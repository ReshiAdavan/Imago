import os
import glob
import argparse
import time
import copy
import numpy as np
import torch
import cv2

# import matplotlib.image as mpimg

from datetime import datetime
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

# from PIL import Image


class FSDataset(Dataset):

    def __init__(self, images, masks, transforms=None):
        """
        images: images to segment
        masks: masks to predict
        transforms: image transformations
        """

        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]

        if self.transforms:
            img = self.transforms(img)
            mask = self.transforms(mask)
            # normalize image
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            img = normalize(img)

        return img, mask


class ConvBlock(nn.Module):
    """convolutional block"""

    def __init__(self, in_c, out_c, args):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=args.ks_convblock, stride=args.stride, padding='same')
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=args.ks_convblock, stride=args.stride, padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_c)
        self.batch_n = args.batch_n

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        if self.batch_n:
            x = self.bn(x)

        x = self.conv2(x)
        x = self.relu(x)
        if self.batch_n:
            x = self.bn(x)

        return x


class EncoderBlock(nn.Module):
    """encoder block"""

    def __init__(self, in_c, out_c, args):
        super(EncoderBlock, self).__init__()

        self.conv = ConvBlock(in_c, out_c, args)
        self.pool = nn.MaxPool2d(args.pool)

    def forward(self, c):
        c = self.conv(c)
        p = self.pool(c)
        return c, p  # return convolutional (c) part for concatenating


class DecoderBlock(nn.Module):
    """
    decoder block
    skip_features:: result from conv block to concatenate
    """

    def __init__(self, in_c, out_c, args):
        super(DecoderBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)  # args.stride)
        self.conv = ConvBlock(in_c, out_c, args)

    def forward(self, x, skip_features):
        x = self.up(x)
        c = torch.cat([x, skip_features], dim=1)
        c = self.conv(c)

        return c


class FSUNet(nn.Module):

    def __init__(self, in_c, out_c, args):
        super(FSUNet, self).__init__()

        self.enc1 = EncoderBlock(args.channels, out_c, args)

        self.enc2 = EncoderBlock(out_c, int(2 * out_c), args)
        self.enc3 = EncoderBlock(int(2 * out_c), int(2 * 2 * out_c), args)
        self.enc4 = EncoderBlock(int(2 * 2 * out_c), int(2 * 2 * 2 * out_c), args)

        self.conv = ConvBlock(int(2 * 2 * 2 * out_c), int(2 * 2 * 2 * 2 * out_c), args)

        self.dec1 = DecoderBlock(int(2 * 2 * 2 * 2 * out_c), int(2 * 2 * 2 * out_c), args)
        self.dec2 = DecoderBlock(int(2 * 2 * 2 * out_c), int(2 * 2 * out_c), args)
        self.dec3 = DecoderBlock(int(2 * 2 * out_c), int(2 * out_c), args)
        self.dec4 = DecoderBlock(int(2 * out_c), out_c, args)

        self.output = nn.Conv2d(out_c, args.classes, kernel_size=1)

    def forward(self, img):
        # input shape (bs, channels, height, width)
        c1, p1 = self.enc1(img)
        c2, p2 = self.enc2(p1)
        c3, p3 = self.enc3(p2)
        c4, p4 = self.enc4(p3)

        b = self.conv(p4)

        x = self.dec1(b, c4)
        x = self.dec2(x, c3)
        x = self.dec3(x, c2)
        x = self.dec4(x, c1)

        x = self.output(x)

        return x


def get_mask_channels(mask, color_codes):
    n_channels = len(color_codes.keys())  # N_CLASSES
    mask_channels = np.zeros((mask.shape[0], mask.shape[1], n_channels),
                             dtype=np.float32)
    for i, cls in enumerate(color_codes.keys()):
        color = color_codes[cls]
        sub_mask = np.all(mask == color, axis=-1) * i
        mask_channels[:, :, i] = sub_mask

    return mask_channels


def get_masks_one_hot(mask, color_codes):
    n_channels = len(color_codes.keys())  # N_CLASSES
    mask_channels = np.zeros((mask.shape[0], mask.shape[1], n_channels),
                             dtype=np.float32)
    for i, cls in enumerate(color_codes.keys()):
        color = color_codes[cls]
        sub_mask = np.all(mask == color, axis=-1) * 1
        mask_channels[:, :, i] = sub_mask
    return mask_channels


def get_preds_one_hot(pred, num_classes, device):
    pred_channels = torch.zeros((pred.size()[0], num_classes, pred.size()[1], pred.size()[2])).to(device)
    for i in range(num_classes):
        sub_pred = (pred == i) * 1
        pred_channels[:, i, :, :] = sub_pred

    return pred_channels


def dice_coef(y_pred, y_true, smooth=1):
    intersection = torch.sum(y_true * y_pred, axis=[1, 2])
    union = torch.sum(y_true, axis=[1, 2]) + torch.sum(y_pred, axis=[1, 2])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    return dice


def dice_coef_multilabel(y_pred, y_true, args):
    dice = 0
    for idx in range(args.classes):
        dice += dice_coef(y_true[:, idx, :, :], y_pred[:, idx, :, :])

    return dice / args.classes


def iou_coef(y_pred, y_true, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred), axis=[1, 2])
    union = torch.sum(y_true, axis=[1, 2]) + torch.sum(y_pred, [1, 2]) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), axis=0)

    return iou


def iou_coef_multilabel(y_pred, y_true, args):
    iou = 0
    for idx in range(args.classes):
        iou += iou_coef(y_true[:, idx, :, :], y_pred[:, idx, :, :])

    return iou / args.classes


def training(model, epoch, device, dataloader, criterion, optimizer, args):
    train_losses = []
    predictions = []
    dice_scores = []
    iou_scores = []

    model.train()

    # loop over batches
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, (img, mask) in loop:
    
        # set to device
        img, mask = img.to(device), mask.to(device)
    
        # set optimizer to zero
        optimizer.zero_grad()
    
        # forward pass (apply model)
        pred = model(img)
    
        # loss
        loss = criterion(pred, mask)
        train_losses.append(loss)
    
        # backward
        loss.backward()
    
        # update the weights
        optimizer.step()

        pred = torch.argmax(pred, axis=1)
        predictions.append(pred)

        pred_one_hot = get_preds_one_hot(pred, args.classes, device)
        dice = dice_coef_multilabel(pred_one_hot, mask, args)
        iou = iou_coef_multilabel(pred_one_hot, mask, args)

        dice_scores.append(dice)
        iou_scores.append(iou)
    
        # update progress bar
        loop.set_description(f'Train Epoch {epoch}/{args.epochs}')
        loop.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    # train loss over all batches
    train_loss = torch.mean(torch.tensor(train_losses))
    train_dice = torch.mean(torch.tensor(dice_scores))
    train_iou = torch.mean(torch.tensor(iou_scores))
    loop.set_postfix(train_loss=train_loss.item(), train_dice=train_dice.item(), train_iou=train_iou.item())

    return train_loss, train_dice, train_iou


def validation(model, epoch, device, dataloader, criterion, args):
    val_losses = []
    predictions = []
    dice_scores = []
    iou_scores = []

    model.eval()
    with torch.no_grad():
    
        # loop over batches
        loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for idx, (img, mask) in loop:
    
            # set to device
            img, mask = img.to(device), mask.to(device)
    
            # forward pass (apply model)
            pred = model(img)
    
            # loss
            loss = criterion(pred, mask)
            val_losses.append(loss)

            pred = torch.argmax(pred, axis=1)
            # predictions.append(pred)

            pred_one_hot = get_preds_one_hot(pred, args.classes, device)
            predictions.append(pred_one_hot)
            dice = dice_coef_multilabel(pred_one_hot, mask, args)
            dice_scores.append(dice)
            iou = iou_coef_multilabel(pred_one_hot, mask, args)
            iou_scores.append(iou)

            loop.set_description(f'Val Epoch {epoch}/{args.epochs}')
            loop.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    # train loss over all batches
    val_loss = torch.mean(torch.tensor(val_losses))
    val_dice = torch.mean(torch.tensor(dice_scores))
    val_iou = torch.mean(torch.tensor(iou_scores))
    loop.set_postfix(val_loss=val_loss.item(), val_dice=val_dice.item(), val_iou=val_iou.item())
    return val_loss, val_dice, val_iou, predictions


def main(args):
    color_codes = {
        'Animal': [64, 128, 64],
        'Archway': [192, 0, 128],
        'Bicyclist': [0, 128, 192],
        # 'Bridge': [0, 128, 64],
        'Building': [128, 0, 0],
        'Car': [64, 0, 128],
        'CartLuggagePram': [64, 0, 192],
        'Child': [192, 128, 64],
        'Column_Pole': [192, 192, 128],
        'Fence': [64, 64, 128],
        'LaneMkgsDriv': [128, 0, 192],
        # 'LaneMkgsNonDriv': [192, 0, 64],
        'Misc_Text': [128, 128, 64],
        # 'MotorcycleScooter': [192, 0, 192],
        'OtherMoving': [128, 64, 64],
        # 'ParkingBlock': [64, 192, 128],
        'Pedestrian': [64, 64, 0],
        'Road': [128, 64, 128],
        # 'RoadShoulder': [128, 128, 192],
        'Sidewalk': [0, 0, 192],
        'SignSymbol': [192, 128, 128],
        'Sky': [128, 128, 128],
        # 'SUVPickupTruck': [64, 128, 192],
        # 'TrafficCone': [0, 0, 64],
        'TrafficLight': [0, 64, 64],
        # 'Train': [192, 64, 128],
        'Tree': [128, 128, 0],
        'Truck_Bus': [192, 128, 192],
        # 'Tunnel': [64, 0, 64],
        'VegetationMisc': [192, 192, 0],
        'Void': [0, 0, 0],
        'Wall': [64, 192, 0]
    }

    # read data
    print('read data...')
    start_time = time.time()
    images_path = glob.glob(f"{args.image_path}/*")
    masks_path = glob.glob(f"{args.mask_path}/*")

    images_path.sort()
    masks_path.sort()

    print(images_path[:5])
    print(masks_path[:5])

    # only load first 100 images and masks in debug mode
    if args.debug:
        images_path = images_path[:10]
        masks_path = masks_path[:10]

    images = []
    masks = []
    for i, img in enumerate(images_path):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_size, args.img_size))
        mask = cv2.imread(masks_path[i])
        mask = cv2.resize(mask, (args.img_size, args.img_size))
        images.append(img)
        masks.append(mask)

        masks_channels = [get_mask_channels(mask, color_codes) for mask in masks]
        masks_one_hot = [get_masks_one_hot(mask, color_codes) for mask in masks]

        # convert_tensor = transforms.ToTensor()
        # arr_img = [convert_tensor(img) for img in images]
        # arr_mask = [convert_tensor(mask) for mask in masks_one_hot]

        # images.append(arr_img)
        # masks.append(arr_mask)

    print(f'reading data took {time.time() - start_time:.2f}s')
    assert len(images) == len(masks_one_hot), 'Nr of images and masks are not the same'
    print(f'data length: {len(images)}')
    print(f'image size: {images[0].shape}, {images[1].shape}, {images[2].shape},...')
    print(f'mask size: {masks[0].shape}, {masks[1].shape}, {masks[2].shape},...')

    # create train and validation data
    imgs_train, imgs_val, masks_train, masks_val = train_test_split(images, masks_one_hot, test_size=0.2,
                                                                    random_state=42)
    print(f'train - validation split created')
    print(f'train data length {len(imgs_train)} - validation data length {len(imgs_val)}')

    # transformations
    transforms_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    transforms_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    # create dataset
    train_dataset = FSDataset(imgs_train, masks_train,
                              transforms=transforms_train)
    val_dataset = FSDataset(imgs_val, masks_val,
                            transforms=transforms_val)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    print(f'train dataloader image batch: {next(iter(train_dataloader))[0].shape}')  # (bs, channels, height, width)
    print(f'train dataloader mask batch: {next(iter(train_dataloader))[1].shape}')
    print(f'val dataloader image batch: {next(iter(val_dataloader))[0].shape}')
    print(f'val dataloader mask batch: {next(iter(val_dataloader))[1].shape}')

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # define model
    in_c = args.channels
    out_c = 64
    model = FSUNet(in_c, out_c, args).to(device)

    # data, target = next(iter(train_dataloader))
    # print(model(data).shape)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define loss
    if args.classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # training and validation
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    train_ious = []
    val_ious = []
    predictions = []
    best_loss = 9999
    best_epoch = -1
    best_model = None

    for epoch in range(args.epochs):

        train_loss, dice_train, iou_train = training(model, epoch, device, train_dataloader, criterion, optimizer, args)
        val_loss, dice_val, iou_val, preds = validation(model, epoch, device, val_dataloader, criterion, args)

        # save model if loss is lower than best_loss
        if args.save and (train_loss < best_loss):
            best_loss = train_loss
            best_epoch = epoch
            now = datetime.now()
            date = now.strftime("%Y%m%d%H%M")
            model_name = f'model_city_{date}_epoch={epoch}_loss={best_loss:.3f}_dice={dice_train:.3f}_iou={iou_train:.3f}.pt'
            model_path = os.path.join(args.save_model_path, model_name)
            preds_name = f'pred_city_{date}_epoch={epoch}_loss={best_loss:.3f}_dice={dice_train:.3f}_iou={iou_train:.3f}'
            preds_path = os.path.join(args.save_preds_path, preds_name)
            # save best model
            best_model = copy.deepcopy(model)
            # save predictions
            best_preds = torch.concat(preds, axis=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(dice_train)
        val_dices.append(dice_val)
        train_ious.append(iou_train)
        val_ious.append(iou_val)
        predictions.append(preds)

    print('train_losses:')
    print(train_losses)
    print('val_losses:')
    print(val_losses)
    print('train dices:')
    print(train_dices)
    print('val dices:')
    print(val_dices)
    print('train ious:')
    print(train_ious)
    print('val ious:')
    print(val_ious)

    # save best model
    if args.save:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss},
            model_path)

        # save predictions on validation set
        # predictions and masks are saved as one-hot encoded arrays

        ## transfer predictions to cpu if necessary and convert to numpy array

        masks_val = np.stack(masks_val, axis=0)
        masks_path = os.path.join(args.save_preds_path, 'mask_val_city')
        if torch.cuda.is_available():
            best_preds = best_preds.cpu().detach().numpy()
        else:
            best_preds = best_preds.detach().numpy()

        ## save predictions as numpy
        print('save predictions and validation masks...')
        np.save(preds_path, best_preds)
        np.save(masks_path, masks_val)
        print(f'predictions saved')
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--image-path', type=str, default='data/forest_segmentation/images')
    parser.add_argument('--mask-path', type=str, default='data/forest_segmentation/masks')
    parser.add_argument('--save-model-path', type=str, default='models')
    parser.add_argument('--save-preds-path', type=str, default='predictions')
    parser.add_argument('--save', action='store_true',
                        default=False)  # if True, best model and predictions are saved o disk
    # data properties
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--classes', type=int, default=23)  # 1 for binary classification
    # train parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pool', type=int, default=2)  # pool size
    parser.add_argument('--ks-convblock', type=int, default=3)  # kernel size of conv block
    parser.add_argument('--stride', type=int, default=1)  # stride size of conv block
    parser.add_argument('--batch_n', action='store_true')  # apply batch norm
    parser.add_argument('--lr', type=float, default=0.001)  # learning rate
    parser.add_argument('--threshold', type=float, default=0.5)  # learning rate
    args = parser.parse_args()

    print('BEGIN argparse key - value pairs')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('END argparse key - value pairs')
    print()

    main(args)

#! /usr/bin/env python3

import os
import glob
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg
from PIL import Image

import u2_net


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


def dice_coef(y_pred, y_true, smooth=1):
    intersection = torch.sum(y_true * y_pred, axis=[1, 2, 3])
    union = torch.sum(y_true, axis=[1, 2, 3]) + torch.sum(y_pred, axis=[1, 2, 3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    return dice


def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred), axis=[1, 2, 3])
    union = torch.sum(y_true, axis=[1, 2, 3]) + torch.sum(y_pred, [1, 2, 3]) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth), axis=0)

    return iou


bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(),
        loss6.data.item()))

    return loss0, loss


def training(model, epoch, device, dataloader, optimizer, args):
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
        print('training')
        print(img.size(), mask.size())
        # set optimizer to zero
        optimizer.zero_grad()
        # forward pass (apply model)
        d0, d1, d2, d3, d4, d5, d6 = model(img)
        pred = d0
        # loss
        # loss = criterion(pred[0], mask)
        loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
        train_losses.append(loss)
        # backward
        loss.backward()
        # update the weights
        optimizer.step()

        pred = (pred > args.threshold).float()
        predictions.append(pred)

        dice = dice_coef(pred, mask)
        iou = iou_coef(pred, mask)

        dice_scores.append(dice)
        iou_scores.append(iou)
        # update progess bar
        loop.set_description(f'Train Epoch {epoch}/{args.epochs}')
        loop.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    # train loss over all batches
    train_loss = torch.mean(torch.tensor(train_losses))
    train_dice = torch.mean(torch.tensor(dice_scores))
    train_iou = torch.mean(torch.tensor(iou_scores))
    loop.set_postfix(train_loss=train_loss.item(), train_dice=train_dice.item(), train_iou=train_iou.item())

    return train_loss, train_dice, train_iou


def validation(model, epoch, device, dataloader, args):
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
            d0, d1, d2, d3, d4, d5, d6 = model(img)
            pred = d0
            # loss
            # loss = criterion(pred[0], mask)
            loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
            val_losses.append(loss)

            pred = (pred > args.threshold).float()
            predictions.append(pred)

            dice = dice_coef(pred, mask)
            dice_scores.append(dice)
            iou = iou_coef(pred, mask)
            iou_scores.append(iou)

            loop.set_description(f'Val Epoch {epoch}/{args.epochs}')
            loop.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())

    # train loss over all batche
    val_loss = torch.mean(torch.tensor(val_losses))
    val_dice = torch.mean(torch.tensor(dice_scores))
    val_iou = torch.mean(torch.tensor(iou_scores))
    loop.set_postfix(val_loss=val_loss.item(), val_dice=val_dice.item(), val_iou=val_iou.item())
    return val_loss, val_dice, val_iou, predictions


def main(args):
    # read data
    print('read data...')
    start_time = time.time()
    images_path = glob.glob(f"{args.image_path}/*.jpg")
    masks_path = glob.glob(f"{args.mask_path}/*.jpg")

    images_path.sort()
    masks_path.sort()

    print(images_path[:5])
    print(masks_path[:5])

    # only load first 100 images and masks in debug mode
    if args.debug:
        images_path = images_path[:100]
        masks_path = masks_path[:100]

    images = []
    masks = []
    for i, img in enumerate(images_path):
        img = Image.open(img)
        if args.classes == 1:
            mask = Image.open(masks_path[i]).convert("L")
        else:
            print('multi class segmentation not implented')
        convert_tensor = transforms.ToTensor()
        arr_img = convert_tensor(img)
        arr_mask = convert_tensor(mask)

        images.append(arr_img)
        masks.append(arr_mask)
        img.close()
        mask.close()

    print(f'reading data took {time.time() - start_time:.2f}s')
    assert len(images) == len(masks), 'Nr of images and masks are not the same'
    print(f'data length: {len(images)}')
    print(f'image size: {images[0].size()}, {images[1].size()}, {images[2].size()},...')
    print(f'mask size: {masks[0].size()}, {masks[1].size()}, {masks[2].size()},...')

    # create train and validation data
    imgs_train, imgs_val, masks_train, masks_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    print(f'train - validation split created')
    print(f'train data length {len(imgs_train)} - validation data length {len(imgs_val)}')

    # transformations
    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
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
    model = u2_net.U2Net(args).to(device)

    # data, target = next(iter(train_dataloader))
    # print(model(data).shape)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define loss
    if args.classes == 1:
        criterion = nn.BCELoss()  # (size_average=True)
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

        train_loss, dice_train, iou_train = training(model, epoch, device, train_dataloader, optimizer, args)
        val_loss, dice_val, iou_val, preds = validation(model, epoch, device, val_dataloader, args)

        # save model if loss is lower than best_loss
        if args.save and (train_loss < best_loss):
            best_loss = train_loss
            best_epoch = epoch
            now = datetime.now()
            date = now.strftime("%Y%m%d%H%M")
            model_name = f'model_forest_{date}_epoch={epoch}_loss={best_loss:.3f}_dice={dice_train:.3f}_iou={iou_train:.3f}.pt'
            model_path = os.path.join(args.save_model_path, model_name)
            preds_name = f'pred_forest_{date}_epoch={epoch}_loss={best_loss:.3f}_dice={dice_train:.3f}_iou={iou_train:.3f}'
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

        ## transfer predictions to cpu if necessary and convert to numpy array

        masks_val = torch.concat(masks_val, axis=0)
        masks_path = os.path.join(args.save_preds_path, 'mask_val_forest')
        if torch.cuda.is_available():
            best_preds = best_preds.cpu().detach().numpy()
            masks_val = masks_val.cpu().detach().numpy()
        else:
            best_preds = best_preds.detach().numpy()
            masks_val = masks_val.detach().numpy()

        ## save predictions as numpy
        print('save predictions and validation masks...')
        np.save(preds_path, best_preds)
        np.save(masks_path, masks_val)
        print(f'predictions saved')

    # TODO
    # inference mode only - load model, make and save predictions


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
    # parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--classes', type=int, default=1)  # 1 for binary classification
    # train parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pool', type=int, default=2)  # pool size
    # parser.add_argument('--ks-convblock', type=int, default=3) # kernel size of conv block
    # parser.add_argument('--stride', type=int, default=1) # stride size of conv block
    # parser.add_argument('--batch_n', action='store_true') # apply batch norm
    parser.add_argument('--lr', type=float, default=0.001)  # learning rate
    parser.add_argument('--threshold', type=float, default=0.5)  # learning rate
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)

    args = parser.parse_args()

    print('BEGIN argparse key - value pairs')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('END argparse key - value pairs')
    print()

    main(args)

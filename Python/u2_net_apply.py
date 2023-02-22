import os
from PIL import Image
import argparse
from torchvision import transforms
import u2_net
from u2_net import U2Net


def main(args):
    file_ = '998002_sat_40.jpg'
    img_path = os.path.join(args.data_dir, file_)
    img = Image.open(img_path)
    print(img.size)

    transform = transforms.ToTensor()
    input_ = transform(img)
    # add batch dimension
    input_ = input_[None, :]
    print(input_.size())

    model = U2Net(args)
    y = model(input_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/forest_segmentation/images')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)
    args = parser.parse_args()

    print('argparse key/value')
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    main(args)

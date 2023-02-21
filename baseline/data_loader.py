from torch.utils.data import DataLoader, Dataset
import random
import torchvision.transforms as transforms
from PIL import Image
import os
from glob import glob


def create_dataloader(args, n_threads=0, is_train=True):
    return DataLoader(
        SRDataset(args, is_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_threads,
        drop_last=False
    )


class SRDataset(Dataset):
    def __init__(self, args, is_train):
        super(SRDataset, self).__init__()
        self.args = args
        self.is_train = is_train
        self.random_trans = transforms.Compose([
            transforms.RandomHorizontalFlip()  # 水平方向随机翻转
        ])
        self.img_trans = self.img_transformer()

        # 读取所有图像路径
        self.img_list = glob(os.path.join(self.args.dataset_path, "**/*.jpg"), recursive=True)

    def __len__(self):
        return len(self.img_list)

    def img_transformer(self):
        return {
            "GT": transforms.Compose([
                transforms.Resize((224, 224)),  # InterpolationMode.BICUBIC
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "LQ": transforms.Compose([
                transforms.Resize((28, 28)),  # InterpolationMode.BICUBIC
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }

    def __getitem__(self, index):
        # 读取原图像
        ori_img = Image.open(self.img_list[index])
        ori_img = ori_img.convert('RGB')

        # 先进行统一的随即变换
        random_img = self.random_trans(ori_img)

        # 再分别变换为低分辨率和高分辨率图像
        lr_img = self.img_trans["LQ"](random_img)
        hr_img = self.img_trans["GT"](random_img)

        return {'LQ': lr_img, 'GT': hr_img}



class SRFERDataset(Dataset):
    def __init__(self, args, is_train):
        super(SRFERDataset, self).__init__()
        self.args = args
        self.is_train = is_train
        self.random_trans = transforms.Compose([
            transforms.RandomHorizontalFlip()  # 水平方向随机翻转
        ])
        self.img_trans = self.img_transformer()

        # 读取所有图像路径
        self.img_list = glob(os.path.join(self.args.dataset_path, "**/*.jpg"), recursive=True)

    def __len__(self):
        return len(self.img_list)

    def img_transformer(self):
        return {
            "GT": transforms.Compose([
                transforms.Resize((224, 224)),  # InterpolationMode.BICUBIC
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "LQ": transforms.Compose([
                transforms.Resize((28, 28)),  # InterpolationMode.BICUBIC
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }

    def __getitem__(self, index):
        # 读取原图像
        ori_img = Image.open(self.img_list[index])
        ori_img = ori_img.convert('RGB')

        # 先进行统一的随即变换
        random_img = self.random_trans(ori_img)

        # 再分别变换为低分辨率和高分辨率图像
        lr_img = self.img_trans["LQ"](random_img)
        hr_img = self.img_trans["GT"](random_img)

        return {'LQ': lr_img, 'GT': hr_img}

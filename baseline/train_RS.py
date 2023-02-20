import os
import math
import argparse
import random
from utils import check_args, display_online_results
from data_loader import create_dataloader
from models.SRGAN_model import SRGANModel
import  matplotlib.pyplot as plt
import torch


def main():
    #### options
    parser = argparse.ArgumentParser()

    # 选择可用的GPU编号
    parser.add_argument('--gpu_ids', type=str, default='0')

    # 配置batch_size
    parser.add_argument('--batch_size', type=int, default=32)

    # Adam优化器参数，G和D分开
    parser.add_argument('--lr_G', type=float, default=1e-4)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, default=0.99)
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--weight_decay_D', type=float, default=0)
    parser.add_argument('--beta1_D', type=float, default=0.9)
    parser.add_argument('--beta2_D', type=float, default=0.99)
    # 学习率调度器
    parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')

    # 迭代总数
    parser.add_argument('--niter', type=int, default=100000)
    # 学习率热身迭代数
    parser.add_argument('--warmup_iter', type=int, default=-1)
    # 学习率更新步长
    parser.add_argument('--lr_steps', type=list, default=[50000])
    # 学习率更新参数gamma
    parser.add_argument('--lr_gamma', type=float, default=0.5)

    # 像素级损失函数计算方式和加权值
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    # 感知损失函数计算方式和加权值
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    # gan损失函数计算方式和加权值
    parser.add_argument('--gan_type', type=str, default='ragan')
    parser.add_argument('--gan_weight', type=float, default=5e-3)

    # 判别器每训练多少次，生成器训练一次
    parser.add_argument('--D_update_ratio', type=int, default=1)
    # 先训练判别器多少次
    parser.add_argument('--D_init_iters', type=int, default=0)

    # 中间结果打印频率
    parser.add_argument('--print_freq', type=int, default=100)
    # 中间结果保存频率
    parser.add_argument('--save_freq', type=int, default=10000)

    # 低分辨率图像尺寸
    parser.add_argument('--lr_size', type=int, default=28)
    # 超分辨率图像尺寸
    parser.add_argument('--hr_size', type=int, default=224)

    # 生成器网络结构参数
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)

    # 判别器网络结构参数
    parser.add_argument('--which_model_D', type=str, default='discriminator_vgg')
    parser.add_argument('--D_in_nc', type=int, default=3)
    parser.add_argument('--D_nf', type=int, default=32)

    # 数据集根目录
    parser.add_argument('--dataset_path', type=str, default='/datasets/rafdb/train/')
    # 检查点存储路径
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/ESRGAN-V1/')
    # 训练状态存储路径，便于resume
    parser.add_argument('--training_state', type=str, default='checkpoints/ESRGAN-V1/state/')

    # 是否resume
    parser.add_argument('--resume_state', type=str, default=None)
    # 加载的G网络预训练权重路径
    parser.add_argument('--pretrain_model_G', type=str, default='pretrain/90000_G.pth')
    # 加载的D网络预训练权重路径
    parser.add_argument('--pretrain_model_D', type=str, default='pretrain/90000_D.pth')

    # 训练配置参数存储路径
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    # 中间结果存储路径
    parser.add_argument('--log_file', type=str, default='log.txt')
    # 解析和检查参数
    args = check_args(parser.parse_args())

    # 指定可用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # 指定使用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_loader = create_dataloader(args, n_threads=8, is_train=True)

    # 创建模型
    model = SRGANModel(args, is_train=True)


    # 是否继续训练
    if args.resume_state is not None:
        # 加载状态字典
        resume_state = torch.load(args.resume_state, map_location=lambda storage, loc: storage.cuda(device))
        # 打印重训练的状态信息
        print('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))
        # 读取重训练的状态信息
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        # 加载优化器和调度器参数
        model.resume_training(resume_state)
        # 加载模型参数
        model.load()
    else:
        current_step = 0
        start_epoch = 0

    # 计算总迭代次数
    total_epochs = int(math.ceil(args.niter / len(train_loader)))

    # 开始训练
    print('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            # step次数加1，起始为1
            current_step += 1
            if current_step > args.niter:
                break

            # 传入当前batch的数据
            model.feed_data(train_data)
            # 训练一个batch并优化参数
            model.optimize_parameters(current_step)

            # 学习率调度器计次加1
            model.update_learning_rate(current_step, warmup_iter=args.warmup_iter)

            # 控制台输出日志信息
            if current_step % args.print_freq == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.6f}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.6f} '.format(k, v)
                print(message)

            # 存储结果
            if current_step % args.save_freq == 0:
                print('Saving models and training states.')
                # 存储网络G和D的权重
                model.save(current_step)
                # 存储训练状态，迭代数、步数、优化器参数、学习率调度器参数
                model.save_training_state(epoch, current_step)

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')


if __name__ == '__main__':
    main()

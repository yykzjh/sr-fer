import os
import math
import argparse
import random
from utils import check_args, pre_write_txt
from data_loader import create_dataloader
from models.SRFER_BaselineModel import SRFERBaselineModel
import torch


def main():
    #### options
    parser = argparse.ArgumentParser()

    # 选择可用的GPU编号
    parser.add_argument('--gpu_ids', type=str, default='0')

    # 数据集根目录
    parser.add_argument('--dataset_path', type=str, default='/datasets/rafdb/')
    # 检查点存储路径
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/SRFER-Baseline/')
    # 训练状态存储路径，便于resume
    parser.add_argument('--training_state', type=str, default='checkpoints/SRFER-Baseline/state/')

    # 低分辨率图像尺寸
    parser.add_argument('--lr_size', type=int, default=28)
    # 超分辨率图像尺寸
    parser.add_argument('--hr_size', type=int, default=224)

    # 配置batch_size
    parser.add_argument('--batch_size', type=int, default=32)

    # 生成器网络结构参数
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)
    # FER网络结构参数
    parser.add_argument('--which_model_FER', type=str, default='ResNet18_children')
    parser.add_argument('--n_classes', type=int, default=7)

    # 像素级损失函数计算方式和加权值
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    # 感知损失函数计算方式和加权值
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    # 生成器G的损失函数整体加权值
    parser.add_argument('--G_weight', type=float, default=0.1)
    # 分类损失函数计算方式和加权值
    parser.add_argument('--fer_criterion', type=str, default='lsr')  # Label Smoothing Regularization(LSR),标签平滑正则化
    parser.add_argument('--FER_weight', type=float, default=1)

    # G的Adam优化器参数
    parser.add_argument('--lr_G', type=float, default=1e-5)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, default=0.99)
    # FER的Adam优化器参数
    parser.add_argument('--lr_FER', type=float, default=0.0003)
    parser.add_argument('--weight_decay_FER', type=float, default=0)
    parser.add_argument('--beta1_FER', type=float, default=0.9)
    parser.add_argument('--beta2_FER', type=float, default=0.99)

    # 学习率调度器
    parser.add_argument('--lr_scheme', type=str, default='CosineAnnealingWarmRestarts')
    # 学习率更新步长
    parser.add_argument('--lr_steps', type=list, default=[250])
    # 学习率更新参数gamma
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    # 学习率更新参数T_max
    parser.add_argument('--lr_T_max', type=int, default=20)
    # 学习率更新参数T_0
    parser.add_argument('--lr_T_0', type=int, default=4)
    # 学习率更新参数T_mult
    parser.add_argument('--lr_T_mult', type=int, default=2)

    # 是否resume
    parser.add_argument('--resume_state', type=str, default=None)
    # 加载的G网络预训练权重路径
    parser.add_argument('--pretrain_model_G', type=str, default='pretrain/latest_G.pth')
    # 加载的FER网络训练权重路径
    parser.add_argument('--pretrain_model_FER', type=str, default='')

    # 迭代总数
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=500)

    # 中间结果打印频率
    parser.add_argument('--print_step_freq', type=int, default=100)
    # 中间结果保存频率
    parser.add_argument('--save_epoch_freq', type=int, default=10)

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
    train_loader = create_dataloader(args, n_threads=8, is_train=True, dataset="SRFER")
    val_loader = create_dataloader(args, n_threads=8, is_train=False, dataset="SRFER")

    # 创建模型
    model = SRFERBaselineModel(args, is_train=True)

    # 是否继续训练
    if args.resume_state is not None:
        # 加载状态字典
        resume_state_dict = torch.load(args.resume_state, map_location=lambda storage, loc: storage.cuda(device))
        # 读取重训练的状态信息
        start_epoch = resume_state_dict['epoch']
        # 打印重训练的状态信息
        print('Resuming training from epoch: {}.'.format(start_epoch))
        # 加载模型参数，包括G、FER、优化器和学习率调整器
        model.load(resume_state_dict)
    else:
        start_epoch = 0
        # 加载模型参数，训练状态只加载G，非训练状态加载G和FER
        model.load()


    # 定义训练用到的变量
    cur_step = 0
    best_accuracy = 0.6
    # 开始训练
    print('Start training from epoch: {:d}, end of epoch: {:d}'.format(start_epoch, args.end_epoch))
    pre_write_txt('Start training from epoch: {:d}, end of epoch: {:d}'.format(start_epoch, args.end_epoch), args.log_file)
    for epoch in range(start_epoch, args.end_epoch):
        # 初始化每个epoch训练部分需要统计的信息
        g_pix_loss = 0.
        g_fea_loss = 0.
        g_loss = 0.
        fer_loss = 0.
        loss = 0.
        train_accuracy = 0.
        batch_cnt = 0

        # 网络调整为训练模式
        model.set_train(mode=True)
        for batch_id, train_data in enumerate(train_loader):

            # 传入当前batch的数据
            model.feed_data(train_data)
            # 训练一个batch并优化参数
            model.optimize_parameters()

            # 获取中间结果
            log = model.get_current_log()
            g_pix_loss += log["g_pix_loss"] * log["cur_batch_size"]
            g_fea_loss += log["g_fea_loss"] * log["cur_batch_size"]
            g_loss += log["g_loss"] * log["cur_batch_size"]
            fer_loss += log["fer_loss"] * log["cur_batch_size"]
            loss += log["loss"] * log["cur_batch_size"]
            train_accuracy += log["step_acc"]
            batch_cnt += log["cur_batch_size"]

            # 控制台输出日志信息
            if (cur_step+1) % args.print_step_freq == 0:
                print("epoch:[{:03d}/{:03d}]  g_pix_loss:{:.6f}  g_fea_loss:{:.6f}  g_loss:{:.6f}  fer_loss:{:.6f}  loss:{:.6f}  train_accuracy:{:.6f}"
                      .format(epoch, args.end_epoch-1, g_pix_loss/batch_cnt, g_fea_loss/batch_cnt, g_loss/batch_cnt,
                              fer_loss/batch_cnt, loss/batch_cnt, train_accuracy/batch_cnt))
                pre_write_txt("epoch:[{:03d}/{:03d}]  g_pix_loss:{:.6f}  g_fea_loss:{:.6f}  g_loss:{:.6f}  fer_loss:{:.6f}  loss:{:.6f}  train_accuracy:{:.6f}"
                              .format(epoch, args.end_epoch-1, g_pix_loss/batch_cnt, g_fea_loss/batch_cnt,
                                      g_loss/batch_cnt, fer_loss/batch_cnt, loss/batch_cnt, train_accuracy/batch_cnt), args.log_file)

            cur_step += 1

        # 计算当前epoch训练部分的中间结果
        g_pix_loss /= batch_cnt
        g_fea_loss /= batch_cnt
        g_loss /= batch_cnt
        fer_loss /= batch_cnt
        loss /= batch_cnt
        train_accuracy /= batch_cnt


        # 初始化每个epoch验证部分需要统计的信息
        batch_cnt = 0
        val_accuracy = 0.
        # 网络调整为验证模式
        model.set_train(mode=False)
        # 不保存梯度信息
        with torch.no_grad():
            for batch_id, val_data in enumerate(val_loader):

                # 传入当前batch的数据
                model.feed_data(val_data)
                # 验证当前batch，并计算正确个数
                model.val_batch()

                # 获取中间结果
                val_accuracy += log["step_acc"]
                batch_cnt += log["cur_batch_size"]

        # 计算当前epoch验证部分的中间结果
        val_accuracy /= batch_cnt


        # 保存模型和训练状态部分
        # 按照一定周期固定保存
        if (epoch+1) % args.save_epoch_freq == 0:
            # 存储网络G和D的权重、训练状态
            model.save(epoch)
        # 每次都保存最新的latest
        model.save(epoch, type="latest")
        # 与最优结果进行比较，保存最优的模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save(epoch, type="best")


        # epoch结束总的输出一下结果
        print("epoch:[{:03d}]  g_pix_loss:{:.6f}  g_fea_loss:{:.6f}  g_loss:{:.6f}  fer_loss:{:.6f}  loss:{:.6f}  train_accuracy:{:.6f}  val_accuracy:{:.6f}  best_accuracy:{:.6f}"
              .format(epoch, g_pix_loss, g_fea_loss, g_loss, fer_loss, loss, train_accuracy, val_accuracy, best_accuracy))
        pre_write_txt("epoch:[{:03d}]  g_pix_loss:{:.6f}  g_fea_loss:{:.6f}  g_loss:{:.6f}  fer_loss:{:.6f}  loss:{:.6f}  train_accuracy:{:.6f}  val_accuracy:{:.6f}  best_accuracy:{:.6f}"
                      .format(epoch, g_pix_loss, g_fea_loss, g_loss, fer_loss, loss, train_accuracy, val_accuracy, best_accuracy), args.log_file)


    print('End of training.')


if __name__ == '__main__':
    main()

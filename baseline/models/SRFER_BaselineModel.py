# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/2/21 0:08
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import logging
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import models.networks as networks
from models.modules.loss import LSR






class SRFERBaselineModel(object):

    def __init__(self, opt, is_train=True):
        super(SRFERBaselineModel, self).__init__()
        # 初始化参数
        self.opt = opt
        self.is_train = is_train
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.schedulers = []
        self.optimizers = []
        self.lr_image = None
        self.hr_image = None
        self.label = None
        self.log_dict = OrderedDict()

        # 初始化生成网络G
        self.netG = networks.define_G(self.opt).to(self.device)
        # 初始化人脸表情识别网络FER
        self.netFER = networks.define_FER(self.opt).to(self.device)

        # 如果用于训练
        if self.is_train:
            # 定义损失函数
            # 定义生成器G相关的损失函数
            # 像素级损失函数
            if self.opt.pixel_weight > 0:
                g_pix_type = self.opt.pixel_criterion
                if g_pix_type == 'l1':
                    self.g_cri_pix = nn.L1Loss().to(self.device)
                elif g_pix_type == 'l2':
                    self.g_cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(g_pix_type))
                self.g_pix_w = self.opt.pixel_weight
            else:
                self.g_cri_pix = None
            # 感知损失函数
            if self.opt.feature_weight > 0:
                g_fea_type = self.opt.feature_criterion
                if g_fea_type == 'l1':
                    self.g_cri_fea = nn.L1Loss().to(self.device)
                elif g_fea_type == 'l2':
                    self.g_cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(g_fea_type))
                self.g_fea_w = self.opt.feature_weight
            else:
                self.g_cri_fea = None
            if self.g_cri_fea:
                # 初始化VGG主干特征提取网络
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            # 初始化G损失函数加权值
            self.g_cri_w = self.opt.G_weight

            # 定义FER相关的损失函数
            if self.opt.FER_weight > 0:
                fer_cri_type = self.opt.fer_criterion
                if fer_cri_type == "lsr":
                    self.fer_cri = LSR(n_classes=self.opt.n_classes, eps=0.1)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(fer_cri_type))
            else:
                self.fer_cri = None
            # 初始化FER损失函数加权值
            self.fer_cri_w = self.opt.FER_weight


            # 定义优化器
            # G
            wd_G = self.opt.weight_decay_G if self.opt.weight_decay_G else 0
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr_G,
                                                weight_decay=wd_G,
                                                betas=(self.opt.beta1_G, self.opt.beta2_G))
            self.optimizers.append(self.optimizer_G)

            # FER
            wd_FER = self.opt.weight_decay_FER if self.opt.weight_decay_FER else 0
            self.optimizer_FER = torch.optim.Adam(self.netFER.parameters(), lr=self.opt.lr_FER,
                                                weight_decay=wd_FER,
                                                betas=(self.opt.beta1_FER, self.opt.beta2_FER))
            self.optimizers.append(self.optimizer_FER)


            # 定义学习率调整器
            if self.opt.lr_scheme == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.opt.lr_steps, gamma=self.opt.lr_gamma)
                    )

            elif self.opt.lr_scheme == "CosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.opt.lr_T_max)
                    )

            elif self.opt.lr_scheme == "CosineAnnealingWarmRestarts":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.opt.lr_T_0, T_mult=self.opt.lr_T_mult)
                    )

            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        # 打印网络参数量
        self.print_network()


    def set_train(self, mode=True):
        self.netG.train(mode=mode)
        self.netFER.train(mode=mode)


    def feed_data(self, data):
        self.lr_image = data["LQ"].to(self.device)
        self.hr_image = data["GT"].to(self.device)
        self.label = data["Label"].to(self.device)
        self.log_dict["cur_batch_size"] = len(self.label)


    def optimize_parameters(self):

        # 网络参数梯度清零
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # G网络前向传播，生成超分辨率图像
        sr_image = self.netG(self.lr_image)
        # 计算G网络的loss
        if self.g_cri_pix:
            g_pix_loss = self.g_cri_pix(sr_image, self.hr_image)
        if self.g_cri_fea:
            hr_image_fea = self.netF(self.hr_image).detach()
            sr_image_fea = self.netF(sr_image)
            g_fea_loss = self.g_cri_fea(sr_image_fea, hr_image_fea)
        # 计算G网络的总loss
        g_loss = self.g_pix_w * g_pix_loss + self.g_fea_w * g_fea_loss

        # FER网络前向传播，对生成的超分辨率图像进行人脸表情识别
        logic_map = self.netFER(sr_image, True)
        # 计算FER网络的loss
        fer_loss = 0
        if self.fer_cri:
            fer_loss = self.fer_cri(logic_map, self.label)

        # 计算总loss
        loss = self.g_cri_w * g_loss + self.fer_cri_w * fer_loss

        # 反向传播
        loss.backward()

        # 优化器对参数进行更新
        for optimizer in self.optimizers:
            optimizer.step()

        # 计算当前batch样本识别正确的个数
        predict_y = torch.max(logic_map, dim=1)[1]
        step_acc = (predict_y == self.label).sum().cpu().item()

        # 记录所有中间结果
        self.log_dict["g_pix_loss"] = g_pix_loss.cpu().item()
        self.log_dict["g_fea_loss"] = g_fea_loss.cpu().item()
        self.log_dict["g_loss"] = g_loss.cpu().item()
        self.log_dict["fer_loss"] = fer_loss.cpu().item()
        self.log_dict["loss"] = loss.cpu().item()
        self.log_dict["step_acc"] = step_acc


    def val_batch(self):
        # G网络前向传播，生成超分辨率图像
        sr_image = self.netG(self.lr_image)
        # FER网络前向传播，对生成的超分辨率图像进行人脸表情识别
        logic_map = self.netFER(sr_image, True)
        # 计算当前batch样本识别正确的个数
        predict_y = torch.max(logic_map, dim=1)[1]
        step_acc = (predict_y == self.label).sum().cpu().item()

        self.log_dict["step_acc"] = step_acc


    def get_current_log(self):
        return self.log_dict


    def print_network(self):
        # G
        n = sum(map(lambda x: x.numel(), self.netG.parameters()))
        print('Network G structure: {}, with parameters: {:,d}'.format(self.netG.__class__.__name__, n))

        # FER
        n = sum(map(lambda x: x.numel(), self.netFER.parameters()))
        print('Network FER structure: {}, with parameters: {:,d}'.format(self.netFER.__class__.__name__, n))

        if self.is_train and self.g_cri_fea:
            n = sum(map(lambda x: x.numel(), self.netF.parameters()))
            print('Network F structure: {}, with parameters: {:,d}'.format(self.netF.__class__.__name__, n))


    def load(self, checkpoint_state_dict=None):
        # 加载G网络参数
        load_path_G = self.opt.pretrain_model_G
        if load_path_G is not None:
            load_net = torch.load(load_path_G, map_location=lambda storage, loc: storage.cuda(self.device))
            self.netG.load_state_dict(load_net, strict=True)
        # 如果是训练模式，需要判断是重头训练还是继续训练
        if self.is_train:
            # 如果是继续训练，则还需要加载FER网络的参数、优化器和学习率调整器的参数
            if self.opt.resume_state is not None:
                # 加载FER网络参数
                load_path_FER = self.opt.pretrain_model_FER
                if load_path_FER is not None:
                    load_net = torch.load(load_path_FER, map_location=lambda storage, loc: storage.cuda(self.device))
                    self.netFER.load_state_dict(load_net, strict=True)
                # 加载训练的状态信息
                if checkpoint_state_dict is not None:
                    # 加载优化器参数
                    resume_optimizers = checkpoint_state_dict['optimizers']
                    assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
                    for i, o in enumerate(resume_optimizers):
                        self.optimizers[i].load_state_dict(o)
                    # 加载学习率调整器参数
                    resume_schedulers = checkpoint_state_dict['schedulers']
                    assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
                    for i, s in enumerate(resume_schedulers):
                        self.schedulers[i].load_state_dict(s)
        else: # 如果不是训练模式，则除了G之外还要加载FER的权重文件
            # 加载FER网络参数
            load_path_FER = self.opt.pretrain_model_FER
            if load_path_FER is not None:
                load_net = torch.load(load_path_FER, map_location=lambda storage, loc: storage.cuda(self.device))
                self.netFER.load_state_dict(load_net, strict=True)
        print("loaded models and training state.")


    def save(self, cur_epoch, type="normal"):
        """
        :param cur_epoch:
        :param type: 可选："normal"|"best"|"latest"
        :return:
        """
        # 获取当前时间
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        # 保存G网络权重文件
        if type == "normal":
            save_filename = '{}_{}_{}.pth'.format(now_str, cur_epoch, "G")
        else:
            save_filename = '{}_{}.pth'.format(type, "G")
        save_path = os.path.join(self.opt.checkpoint_dir, save_filename)
        state_dict = self.netG.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        # 保存FER网络权重文件
        if type == "normal":
            save_filename = '{}_{}_{}.pth'.format(now_str, cur_epoch, "FER")
        else:
            save_filename = '{}_{}.pth'.format(type, "FER")
        save_path = os.path.join(self.opt.checkpoint_dir, save_filename)
        state_dict = self.netFER.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        # 保存当前训练状态的状态字典
        state = {'epoch': cur_epoch, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        if type == "normal":
            save_filename = '{}_{}.state'.format(now_str, cur_epoch)
        else:
            save_filename = '{}.state'.format(type)
        save_path = os.path.join(self.opt.training_state, save_filename)
        torch.save(state, save_path)
        print("saved models and training state.")




























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
import torch.autograd as autograd
import models.networks as networks
from models.modules.loss import LSR






class SRFERIdeaModel(object):

    def __init__(self, opt, is_train=True):
        super(SRFERIdeaModel, self).__init__()
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
            # 定义FER的GradCAM的损失函数
            if self.opt.GradCAM_weight > 0:
                GradCAM_cri_type = self.opt.GradCAM_criterion
                # 定义GradCAM的特征图的损失值的计算方式
                if GradCAM_cri_type == 'l1':
                    self.GradCAM_cri = nn.L1Loss().to(self.device)
                elif GradCAM_cri_type == "l2":
                    self.GradCAM_cri = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(GradCAM_cri_type))
                self.GradCAM_w = self.opt.GradCAM_weight
            else:
                self.GradCAM_cri = None
            # 定义计算GradCAM损失值用到的网络
            if self.GradCAM_cri:
                # 初始化已经与训练好的FER网络
                self.netGradCAMFER = networks.define_FER(self.opt).to(self.device)
                self.netGradCAMFER.load_state_dict(torch.load(self.opt.GradCAM_pretrain_model_FER))
                self.netGradCAMFER.eval()

            # 定义FER的分类相关的损失函数
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




    def cal_GradCAM(self, x, netFER, net_is_pretrain=False):
        """
        计算输入的FER网络的各层的类激活图
        :param x: 从SR网络输出的超分辨率特征图
        :param netFER: FER网络，可以是预训练好的FER，也可能是正在训练的FER
        :param net_is_pretrain: 传入的网络是不是预训练好的网络
        :return: 列表和网络输出，存储各层的类激活图heatmap
        """

        # 清空梯度
        netFER.zero_grad()
        # 先初始化存储梯度的列表
        grad_list = []
        # 初始化存储钩子的列表，用于释放钩子
        hook_list = []
        # 初始化保存中间特征图的列表
        feature_list = []

        # 先解析输入网络的各层，便于分部计算设定钩子
        children_layers = list(netFER.children())
        assert len(children_layers) == 7, "submodules length error"

        out = x
        # 循环执行各层的运算，在指定位置注册钩子
        for i, layer in enumerate(children_layers):
            # 先运算
            out = layer(out)
            # 如果是前五层的卷积层, 需要注册钩子
            if i < 5:
                # 注册钩子
                hook_list.append(out.register_hook(lambda grad: grad_list.append(grad)))
                # 保存特征图
                feature_list.append(out)
            if i == 5:
                out = torch.flatten(out, 1)

        # 提取out中最大类别的输出数值
        pred_index = torch.argmax(out, 1)
        pred_value = out.gather(dim=-1, index=pred_index.unsqueeze(1)).squeeze(1)

        # 反向传播到x,获取中间参数的梯度
        if net_is_pretrain:
            _ = autograd.grad(pred_value, x, grad_outputs=torch.ones_like(pred_value), retain_graph=False)[0]
        else:
            _ = autograd.grad(pred_value, x, grad_outputs=torch.ones_like(pred_value), retain_graph=True, create_graph=True)[0]

        # 定义输出的类激活图列表
        heatmaps = []
        # 先判断得到的梯度和特征图个数是否一致
        assert len(grad_list) == len(feature_list), "grad and feature length error"
        # 循环遍历所有的梯度和对应的特征图
        feature_list_len = len(feature_list)
        for i in range(len(feature_list)):
            # 计算梯度的通道分数
            pool_grad = torch.nn.functional.adaptive_avg_pool2d(grad_list[feature_list_len - 1 - i], (1, 1))
            # 分数张量和特征图相乘得到类激活图
            heatmap = pool_grad * feature_list[i]

            # 类激活图的通道维度取平均
            heatmap = torch.mean(heatmap, dim=1, keepdim=True)
            # 类激活图ReLU
            nn.ReLU(inplace=True)(heatmap)
            # 类激活图归一化
            max_heatmap = torch.nn.AdaptiveMaxPool2d((1, 1))(heatmap)
            min_heatmap, _ = torch.min(heatmap.view(heatmap.size(0), -1), dim=-1)
            min_heatmap = min_heatmap.view(-1, 1, 1, 1)
            heatmap = (heatmap - min_heatmap) / (max_heatmap - min_heatmap + 1e-38)
            if torch.isnan(heatmap).any():
                print(heatmap)
            # 保存
            heatmaps.append(heatmap.squeeze(1))

        # 注销钩子
        for hook in hook_list:
            hook.remove()
        # 清空梯度
        netFER.zero_grad()

        return heatmaps, out


    def optimize_parameters(self):

        # 网络参数梯度清零
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # G网络前向传播，生成超分辨率图像
        sr_image = self.netG(self.lr_image)
        # GradCAMFER前向传播，计算类激活图
        GradCAMFER_heatmaps, _ = self.cal_GradCAM(sr_image, self.netGradCAMFER, net_is_pretrain=True)
        # FER网络前向传播，并计算类激活图
        FER_heatmaps, logic_map = self.cal_GradCAM(sr_image, self.netFER, net_is_pretrain=False)

        # 计算类激活图loss
        GradCAM_loss_list = []
        # 判断两个网络输出的类激活图是否一致
        assert len(FER_heatmaps) == len(GradCAMFER_heatmaps), "heatmaps length error"
        # 遍历计算l2 loss
        for i, FER_heatmap in enumerate(FER_heatmaps):
            GradCAM_loss_list.append(self.GradCAM_cri(FER_heatmap, GradCAMFER_heatmaps[i]))
            # print(self.GradCAM_cri(FER_heatmap, GradCAMFER_heatmaps[i]))
        # 计算GradCAM loss均值
        GradCAM_loss = torch.FloatTensor(GradCAM_loss_list).to(self.device).mean()

        # 计算FER网络的loss
        fer_loss = 0
        if self.fer_cri:
            fer_loss = self.fer_cri(logic_map, self.label)

        # 计算总loss
        loss = self.GradCAM_w * GradCAM_loss + self.fer_cri_w * fer_loss

        # 反向传播
        loss.backward()

        # 优化器对参数进行更新
        for optimizer in self.optimizers:
            optimizer.step()

        # 计算当前batch样本识别正确的个数
        predict_y = torch.max(logic_map, dim=1)[1]
        step_acc = (predict_y == self.label).sum().cpu().item()

        # 记录所有中间结果
        self.log_dict["GradCAM_loss"] = GradCAM_loss.cpu().item()
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

        if self.is_train and self.GradCAM_cri:
            n = sum(map(lambda x: x.numel(), self.netGradCAMFER.parameters()))
            print('Network GradCAMFER structure: {}, with parameters: {:,d}'.format(self.netGradCAMFER.__class__.__name__, n))


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




























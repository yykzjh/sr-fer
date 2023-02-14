import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import sys
from torch.nn import functional as F
# sys.path.append('./')
# from model import AlexNet
# from loss import Regularization
# from vit_model import VisionTransformer
import data_input
import os,sys
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '.'))
# import resnet
# from model import TS_model
# from model3 import ResNet50 
from model3 import ResNet18
from model3 import ResNet18_children


import os
import json
import time

import torchvision.models as models

def seed_torch(seed=66):
    import random
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class LSR(nn.Module):
    def __init__(self, n_classes=7, eps=0.1):
        super(LSR, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, outputs, labels):
        # labels.shape: [b,]
        assert outputs.size(0) == labels.size(0)
        n_classes = self.n_classes
        one_hot = F.one_hot(labels, n_classes).float()
        mask = ~(one_hot > 0)
        smooth_labels = torch.masked_fill(one_hot, mask, self.eps / (n_classes - 1))
        smooth_labels = torch.masked_fill(smooth_labels, ~mask, 1 - self.eps)
        ce_loss = torch.sum(-smooth_labels * F.log_softmax(outputs, 1), dim=1).mean()
        # ce_loss = F.nll_loss(F.log_softmax(outputs, 1), labels, reduction='mean')
        return ce_loss

LSR_loss = LSR(n_classes=7, eps=0.1)

# data_dir = '/datasets/AffectNet/' /datasets/RAFDB/compound/Image/
# data_dir = '/datasets/RAFDB/compound/Image/'
# data_dir = '../rafdb/'
data_dir = '/datasets/rafdb/'
# data_dir = '/home/xie/ferplus/'
batch_size = 256
# trainset loader
train_loader, tra_num = data_input.train_data(data_dir,batch_size)
# print(train_loader)
# validation loader
validate_loader, val_num =  data_input.val_data(data_dir,128)

# net = AlexNet(num_classes=7, init_weights=True)
# net = ResNet18(num_classes=7)

# models.resnet50(pretrained=True)
# net = TS_model()
net = ResNet18_children(num_classes=7)
# net = ResNet50()
# net.init_weights()
# num_features=net.fc.in_features
# net.fc=nn.Linear(num_features,7)
# net.load_state_dict(torch.load('resnet18-5c106cde.pth'))
# net.fc = nn.Linear(2048, 7) 



pre_train_model = 'AlexNet3_vit.pth'
save_path = './AlexNet3_vit2.pth'

is_recovery = False
if is_recovery:
    checkpoint = torch.load(pre_train_model)
    net.load_state_dict(checkpoint)
    print("-------- 模型恢复训练成功-----------")
    save_path = './pretrain_{}_.pth'.format(pre_train_model.split('.')[0])

# num_features = net.fc.in_features
# net.fc = nn.Linear(num_features, 11)
net.to(device)
# weght_ce = torch.tensor([0.0939, 0.4310, 0.1689, 0.0254, 0.0611, 0.1718, 0.0480], dtype=torch.float32).to(device)
# weight=weght_ce
loss_function = nn.CrossEntropyLoss()
pata = list(net.parameters())  # 查看net内的参数 lr 0.0001
optimizer = optim.Adam(net.parameters(), lr=0.0003)

# 写入文本
def pre_write_txt(pred, file):
    f = open(file, 'a', encoding='utf-8')
    f.write(str(pred))
    f.write('\n')
    f.close()
    print("-----------------预测结果已经写入文本文件--------------------")


# scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min=0, last_epoch=-1)
import random
best_acc = 0.0
epoch_test_acc =0.9
# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}

#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()

#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()

#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name]

#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}

# 初始化
# ema = EMA(net, 0.999)
# ema.register()

for epoch in range(80):
   
    # train
    net.train()  # 在训练过程中调用dropout方法
    running_loss = 0.0
    t1 = time.perf_counter()  # 统计训练一个epoch所需时间
    # print('star')
    tra_acc = 0.0
    # torch.autograd.set_detect_anomaly(True)
    for step, data in enumerate(train_loader, start=0):
        # 这个时候的标签还是数字 不是onehot
        # print(step)
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # if epoch < 0:
        #    if epoch_test_acc < 0.88:
        #      index = [i for i in range(len(labels))]
        #      random.shuffle(index)
        #      labels = labels[index]
        # print(labels)
        optimizer.zero_grad()
        # alpha=0.1
        # lam = np.random.beta(alpha,alpha)
        # index = torch.randperm(images.size(0)).cuda()
        # inputs = lam*images + (1-lam)*images[index,:]
        # targets_a, targets_b = labels, labels[index]

        outputs = net(images,True)
        # loss = lam * loss_function(outputs, targets_a) + (1 - lam) * loss_function(outputs, targets_b)
        # print(type(outputs))
        # print(type(labels))
        # out= outputs LSR_loss(outputs, labels)
        # loss = loss_function(outputs, labels)
        loss = LSR_loss(outputs, labels)
        
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()
        # ema.update()


        # print statistics
        tra_predict_y = torch.max(outputs, dim=1)[1]
        step_acc = (tra_predict_y == labels.to(device)).sum().item()
        tra_acc += step_acc
        running_loss += loss.item()
        # each 10 step(or batch) print once
        if (step+1)%10 == 0:
            print("step:{} train acc:{:.3f} train loss:{:.3f}".format(step,step_acc/len(labels),loss))

    scheduler.step()
    one_epoch_time = time.perf_counter()-t1


    # validate
    # ema.apply_shadow()
    net.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
    # ema.restore()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        
        for data_test in validate_loader:
            test_images, test_labels = data_test
            test_labels_len = len(test_labels)
            outputs = net(test_images.to(device),False)
            out= outputs
            predict_y = torch.max(out, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()

        accurate_test = acc / val_num
        epoch_test_acc = accurate_test
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
            if 0.91 < accurate_test < 0.98:
                torch.save(net.state_dict(), './AlexNet_{:.3f}.pth'.format(accurate_test))
        print('\n[epoch %d] trainset_acc:%.3f train_loss: %.3f  testset_accuracy: %.3f best_acc: %.3f one_epoch_time:%.3fs\n' %
              (epoch + 1, tra_acc/tra_num, running_loss / step, accurate_test,best_acc,one_epoch_time))
        pre_write_txt("epoch:{} trainset_acc:{:.3f} train_loss:{:.3f} testset_accuracy: {:.3f} best_acc: {:.3f}".format(epoch + 1, tra_acc/tra_num, running_loss / step, accurate_test,best_acc), file = 'result_vit.txt')

print('Finished Training')


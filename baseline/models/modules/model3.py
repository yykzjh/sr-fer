from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
# from transformer import TBAM
# from vit_sp import TBAM
# from new_vit import TBAM
# from new_vit import Cls

class ResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0.0):
        super(ResNet50, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)


    def forward(self, x,is_training=False):
        x = self.features(x)

        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out

class ResNet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0.0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        checkpoint = torch.load('./resnet18_msceleb.pth') 
        resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features
        self.fc = nn.Linear(num_features, num_classes)


    def forward(self, x,is_training=False):
        x = self.features(x)
        # print(x.size())

        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out



class ResNet18_children(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0.0):
        super(ResNet18_children, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        checkpoint = torch.load('./pretrain/resnet18_msceleb.pth')
        resnet.load_state_dict(checkpoint['state_dict'],strict=True)
        # # self.features = nn.Sequential(*list(resnet.children())[:-2])  # before avgpool 512x1
        # # #--------------
        # # # num_features = resnet.fc.in_features
        # # # resnet18.fc = nn.Linear(num_features,7)
        # # # resnet18.fc = nn.Linear(512,7)
        children = list(resnet.children())

        # self.pre_conv = nn.Sequential(*children[0:4])
        self.pre_conv = nn.Sequential(*children[0:4])
        self.res_block1 = children[4]
        self.res_block2 = children[5]
        self.res_block3 = children[6]
        self.res_block4 = children[7]

        
        self.gloabl_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.arm = Amend_raf2()
        # self.fc0 = nn.Linear(25088, 7)
        
        self.fc = nn.Linear(512, num_classes)
        
       
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight.data)


    def forward(self, x,is_training=False):
        x_pre = self.pre_conv(x)

        x = self.res_block1(x_pre)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.gloabl_avg_pool(x)
        # print("-------x {} ".format(x.size()))
        # 
        
        x = torch.flatten(x, 1)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)

        x = self.fc(x)


        return x


if __name__ == '__main__':
    net = ResNet18()
    x = torch.randn(1,3,224,224)
    out = net(x)
    print(out.size())
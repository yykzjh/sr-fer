from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
# from transformer import TBAM
# from vit_sp import TBAM
from new_vit import TBAM
from new_vit import Cls

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, num_classes=7, drop_rate=0.3):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.features(x)

        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out, out



class ResNet18_ARM___RAF(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(ResNet18_ARM___RAF, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        # # self.features = nn.Sequential(*list(resnet.children())[:-2])  # before avgpool 512x1
        # # #--------------
        # # # num_features = resnet.fc.in_features
        # # # resnet18.fc = nn.Linear(num_features,7)
        # # # resnet18.fc = nn.Linear(512,7)
        children = list(resnet.children())

        self.pre_conv = nn.Sequential(*children[0:4])
        
        self.block1_1 = children[4][0] 
        self.block1_2 = children[4][1]
        
        self.block2_1 = children[5][0]
        self.block2_2 = children[5][1]
        # self.att2 = ...
        self.block3_1 = children[6][0]
        self.block3_2 = children[6][1]
        # self.att3 = ...
        self.block4_1 = children[7][0]
        self.block4_2 = children[7][1]


        # self.arrangement = nn.PixelShuffle(16)
        self.arm = Amend_raf3()
        self.gloabl_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # self.arm = Amend_raf2()
        # self.fc0 = nn.Linear(25088, 7)
        
        self.fc = nn.Linear(512*7*7, num_classes)
        self.TBAM1 = TBAM(ch_embed_dim=56*56,num_heads=3,ch_num_patchs=64, ffn_dim=128,patch_size=8)
        self.TBAM1_1 = TBAM(ch_embed_dim=56*56,num_heads=3,ch_num_patchs=64, ffn_dim=128,patch_size=8)

        self.TBAM2 = TBAM(ch_embed_dim=28*28,num_heads=3,ch_num_patchs=128,ffn_dim=128,patch_size=4)
        self.TBAM2_1 = TBAM(ch_embed_dim=28*28,num_heads=3,ch_num_patchs=128,ffn_dim=128,patch_size=4)

        self.TBAM3 = TBAM(ch_embed_dim=14*14,num_heads=3,ch_num_patchs=256,ffn_dim=128,patch_size=7)
        self.TBAM3_1 = TBAM(ch_embed_dim=14*14,num_heads=3,ch_num_patchs=256,ffn_dim=128,patch_size=2)

        self.TBAM4 = TBAM(ch_embed_dim=7*7,num_heads=3,ch_num_patchs=512,ffn_dim=128,patch_size=7)
        self.TBAM4_1 = TBAM(ch_embed_dim=7*7,num_heads=3,ch_num_patchs=512,ffn_dim=128,patch_size=1)
        # self.TBAM4_2 = TBAM(ch_embed_dim=14*14,num_heads=3,ch_num_patchs=512,ffn_dim=128,patch_size=2)
        # self.TBAM4_3 = TBAM(ch_embed_dim=14*14,num_heads=3,ch_num_patchs=512,ffn_dim=128,patch_size=2)
        # self.TBAM4_4 = TBAM(ch_embed_dim=14*14,num_heads=3,ch_num_patchs=512,ffn_dim=128,patch_size=2)
        # self.TBAM4_4 = TBAM(ch_embed_dim=14*14,num_heads=3,ch_num_patchs=512,ffn_dim=128,patch_size=2)
        self.v = Cls(
                image_size = 7,
                patch_size = 1,
                num_classes = 7,
                dim = 512,
                depth = 3,
                heads = 3,
                mlp_dim = 512,
                channels = 512,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        # self.bn1 = nn.BatchNorm2d(512)
        # self.bn2 = nn.BatchNorm2d(512)
        # self.bn3 = nn.BatchNorm2d(512)
        # self.bn4 = nn.BatchNorm2d(512)
        

        self.alpha1 = nn.Parameter(torch.tensor([1.0]))
        self.alpha2 = nn.Parameter(torch.tensor([1.0]))
        self.alpha3 = nn.Parameter(torch.tensor([1.0]))
        self.alpha4 = nn.Parameter(torch.tensor([1.0]))
        self.cnn = nn.Conv2d(in_channels=512, out_channels=7, kernel_size=1, stride=1, padding=0, bias=True)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)


    def forward(self, x,is_training=False):
        x_pre = self.pre_conv(x)
        x_block1 = self.block1_1(x_pre)
        # print(x_block1.size())
        x_tb1 = self.TBAM1(x_block1)
        # print(x_tb1.size())
        x_block1_2 = self.block1_2(x_tb1)
        
        x_res1 = self.TBAM1_1(x_block1_2)
        # print(x_res1.size())

        x_block2_1 = self.block2_1(x_res1)
        x_tb2 = self.TBAM2(x_block2_1)
        x_block2_2 = self.block2_2(x_tb2)
        # x_res2 = x_block2_2
        x_res2 = self.TBAM2_1(x_block2_2)

        x_block3_1 = self.block3_1(x_res2)
        x_tb3 = self.TBAM3(x_block3_1)
        x_block3_2 = self.block3_2(x_tb3)
        # x_res3 = x_block3_2
        x_res3 = self.TBAM3_1(x_block3_2)

        x_block4_1 = self.block4_1(x_res3)
        x_tb4 = self.TBAM4(x_block4_1)
        x_block4_2 = self.block4_2(x_tb4)
        # x_res4 = x_block4_2
        x_out = self.TBAM4_1(x_block4_2)
        

        # x_out = self.gloabl_avg_pool(x_res4)
        # print("-------x_out {} ".format(x_out.size()))
        x_out = self.arm(x_out,is_training)
        
        x_out = torch.flatten(x_out, 1)

        if self.drop_rate > 0:
            x_out = nn.Dropout(self.drop_rate)(x_out)

        x_out = self.fc(x_out)


        # x_out = x_out
        # print("------- {} ".format(x_out.size()))

        return x_out

class Amend_raf(nn.Module):  # moren
    def __init__(self, inplace=2):
        super(Amend_raf, self).__init__()
        self.bn = nn.BatchNorm2d(inplace)
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.de_albino = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=32, stride=8, padding=0, bias=False)

    def forward(self, x):
        # print(x.size())
        mask = torch.tensor([]).cuda()
        createVar = locals()
        for i in range(x.size(1)):
            createVar['x' + str(i)] = torch.unsqueeze(x[:, i], 1)
            createVar['x' + str(i)] = self.de_albino(createVar['x' + str(i)])
            # print(x.size())
            mask = torch.cat((mask, createVar['x' + str(i)]), 1)
        x = self.bn(mask)
        # print(x.size())
        # xmax, _ = torch.max(x, 1, keepdim=True)
        global_mean = x.mean(dim=[0, 1])
        xmean = torch.mean(x, 1, keepdim=True)
        # xmin, _ = torch.min(x, 1, keepdim=True)
        x = xmean + self.alpha * global_mean

        return x, self.alpha

class Amend_raf2(nn.Module):  # moren
    def __init__(self, inplace=1):
        super(Amend_raf2, self).__init__()
        # self.de_albino = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=32, stride=8, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(inplace)
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        mask = 0
        createVar = locals()

        # for i in range(x.size(1)):
        #     createVar['x' + str(i)] = torch.unsqueeze(x[:, i], 1)
        #     # createVar['x' + str(i)] = self.de_albino(createVar['x' + str(i)])
        #     # print("-------mask {} ".format(mask.size()))
        #     # print("-------createVar['x' + str({})] {} ".format(i,createVar['x' + str(i)].size()))

        #     mask = mask + createVar['x' + str(i)]
        # print("-------mask {} ".format(mask.size()))
        # x = self.bn(mask)
        # x = mask
        # mask = torch.mean(x, 1, keepdim=True)
        # print("-------mask {} ".format(mask.size()))
        # mask = torch.sigmoid(mask)
        # print("-------mask {} ".format(mask.size()))
        
        # xmax, _ = torch.max(x, 1, keepdim=True)
        
        xmean = torch.mean(x, 1, keepdim=True)
        # xmin, _ = torch.min(x, 1, keepdim=True)
        # x = mask*xmean
        # global_mean = xmean.mean(dim=[0])
        # global_mean = torch.sigmoid(global_mean)
        x = x+xmean

        # print("-------x {} ".format(x.size()))

        # x = mask

        return x, self.alpha


# each batch index add operation
class Amend_raf3(nn.Module):  # moren
    def __init__(self, inplace=512):
        super(Amend_raf3, self).__init__()
        # self.de_albino = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=32, stride=8, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(inplace)
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.register_buffer("global_mean",torch.torch.zeros([1,512,7,7]))
        # self.register_buffer("global_mean2",torch.torch.zeros([1,512,7,7]))
        # self.decay = 0.1

    def forward(self, x,is_training):
        if is_training:
            x_mean = x.mean(dim=0,keepdim=True)
            # method 1
            # self.global_mean = (1-self.alpha)*x_mean + self.alpha*self.global_mean.detach()
            # method2 
            self.global_mean = self.alpha*x_mean
            x = x * torch.sigmoid(self.global_mean) + self.global_mean
            x = x + self.global_mean
        else:
            x = x*torch.sigmoid(self.global_mean) + self.global_mean
            # x = x + self.global_mean

        return x




# if __name__=='__main__':
#     model = ResNet18_ARM___RAF()
#     model.cuda()
#     input = torch.randn(1, 3, 224, 224)
#     out, alpha = model(input.cuda())
#     print(out.size())

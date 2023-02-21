import math
import torch
import logging
import models.modules.discriminator_vgg_arch as Discriminator_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.model3 as FER_arch

logger = logging.getLogger('base')


####################
# define network
####################
### FER
def define_FER(opt):
    opt_net = opt
    which_model = opt_net.which_model_FER

    if which_model == "ResNet18_children":
        netFER = FER_arch.ResNet18_children(num_classes=opt.n_classes)
    else:
        raise NotImplementedError('FER model [{:s}] not recognized'.format(which_model))
    return netFER


#### Generator
def define_G(opt):
    opt_net = opt
    which_model = opt_net.which_model_G
    up_ratio = int(math.log(opt.hr_size / opt.lr_size, 2))

    if which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net.G_in_nc, out_nc=opt_net.out_nc,
                                    nf=opt_net.G_nf, nb=opt_net.nb, up_ratio=up_ratio)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt
    which_model = opt_net.which_model_D

    if which_model == 'discriminator_vgg':
        netD = Discriminator_arch.Discriminator_VGG(in_nc=opt_net.D_in_nc, nf=opt_net.D_nf, in_size=opt.hr_size)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt.gpu_ids
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = Discriminator_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

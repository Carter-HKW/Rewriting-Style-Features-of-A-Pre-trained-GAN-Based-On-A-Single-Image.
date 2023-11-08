import sys
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms as transforms
import util.util as util
sys.path.append('./transfer/CAST_pytorch-main')
from models import net



class CAST() :
    def __init__(self) :
        
        self.device = torch.device('cuda')
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        vgg = net.vgg
        vgg.load_state_dict(torch.load('./transfer/CAST_pytorch-main/models/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31]) 
        self.netAE = net.ADAIN_Encoder(vgg)
        self.netDec_A = net.Decoder()
        self.netDec_B = net.Decoder()  
        init_net(self.netAE, 'normal', 0.02)  
        init_net(self.netDec_A, 'normal', 0.02)  
        init_net(self.netDec_B, 'normal', 0.02)
        self.model_names = ['AE','Dec_A', 'Dec_B']

    def forward(self, source) :
        
        transform = get_transform()

        #transfer tensor to img
        total_A = []
        
        for img in source :
            # img = torch.unsqueeze(img, dim=0)
            # print(img.shape)
            im = util.tensor2im(img)
            
            image_pil = Image.fromarray(im)
            A = transform(image_pil)
            # p = os.path.join('./checkpoint/', 'experiment')
            # p = os.path.join(p, 'check')
            # img_path = os.path.join(p, 'iter:%d.png' %i)
            # i += 1
            # temp = util.tensor2im(A)
            # t = Image.fromarray(temp)
            # t.save(img_path)
            A = torch.unsqueeze(A, dim=0)

            total_A.append(A)
        
        A = torch.cat(total_A, dim=0)

        Bimg = Image.open('./transfer/style_data/impress.jpg').convert('RGB')
        B = transform(Bimg)
        B = torch.unsqueeze(B, dim=0)
        B = B.expand(A.shape[0], -1, -1, -1)

        real_A = A.to(self.device)
        real_B = B.to(self.device)
        
        real_A_feat = self.netAE(real_A, real_B)  # G_A(A)
        fake_B = self.netDec_B(real_A_feat)
        # im = util.tensor2im(fake_B)
        # image_pil = Image.fromarray(im)
        
        # image_pil.save("/home/cglab126/cglab205/code/result/put", 'png')
        return fake_B, real_B

    def setup(self):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        model_names = self.model_names
        for name in model_names:
                if isinstance(name, str):
                    load_filename = '%s_net_%s.pth' % ('latest', name)
                    
                    load_dir = './transfer/CAST_pytorch-main/CAST_model'

                    load_path = os.path.join(load_dir, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict)
        

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def get_transform(method=transforms.InterpolationMode.BICUBIC):
    transform_list = []

    osize = [256, 256]
    transform_list.append(transforms.Resize(osize, method))
    transform_list.append(transforms.RandomCrop(256))

    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))





    transform_list += [transforms.ToTensor()]

    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)

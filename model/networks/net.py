import torch.nn as nn
import torch
from torchvision.ops import FeaturePyramidNetwork

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class vgg_loss() :
    def __init__(self, opt) :
        self.device = 'cuda' if opt.gpus > 0 else 'cpu'
        self.vgg = vgg.to(self.device)
        self.vgg.load_state_dict(torch.load(opt.vgg))

        enc_layers = list(self.vgg.children())[:44]
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        self.enc_6 = nn.Sequential(*enc_layers[31:34])  # relu4_1 -> relu4_2
        self.enc_7 = nn.Sequential(*enc_layers[44:47])  # relu5_1 -> relu5_2
        self.enc_8 = nn.Sequential(*enc_layers[:44])
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def encode2feat(self, source) :
        result = [source]
        for i in range(5) :
            func = getattr(self, 'enc_%d' %(i + 1))
            result.append(func(result[-1]))
        return result[1:]
    
    def cal_mean_std(self, feat, eps=1e-5) :
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


    def cal_loss_style(self, output, target) :
        output_mean, output_std = self.cal_mean_std(output)
        target_mean, target_std = self.cal_mean_std(target)
        return self.mse(output_mean, target_mean) + self.mse(output_std, target_std)
    
    def cal_loss_content(self, output, target, norm=False) :
        if norm :
            output_mean, output_std = self.cal_mean_std(output)
            output_norm = (output - output_mean.expand(output.size()) / output_std.expand(output.size()))

            target_mean, target_std = self.cal_mean_std(target)
            target_norm = (target - target_mean.expand(target.size()) / target_std.expand(target.size()))
            return self.mse(output_norm, target_norm) 

        else :
            return self.mse(output, target)
    
    def get_S_loss(self, output, target) :
        output_feat = self.encode2feat(output)
        target_feat = self.encode2feat(target)
        loss_style = 0
        for i in range(0, 5) :
            # loss_style += self.mse(output_feat[i], target_feat[i])
            loss_style += self.cal_loss_style(output_feat[i], target_feat[i])
            # output_gram = self.gram_mat(output_feat[i])
            # target_gram = self.gram_mat(target_feat[i])
            # loss_style += self.mse(output_gram, target_gram) 
        return loss_style 
    def get_C_loss(self, output, target) :
        
        output_feat = self.encode2feat(output)
        target_feat = self.encode2feat(target)
        # output_feat1 = self.enc_6(output_feat[3])
        # output_feat2 = self.enc_7(output_feat[4])
        # target_feat1 = self.enc_6(target_feat[3])
        # target_feat2 = self.enc_7(target_feat[4])
        loss_content = self.mse(output_feat[3], target_feat[3]) + self.mse(output_feat[4], target_feat[4]) 
        # loss_content = self.cal_loss_content(output_feat[3], target_feat[3], norm=True) + self.cal_loss_content(output_feat[4], target_feat[4], norm=True) 
        #
        return loss_content 
    def get_fpn_loss(self, output, target) :
        out_result = {0 : output}
        for i in range(5) :
            func = getattr(self, 'enc_%d' %(i + 1))
            out_result[i + 1] = func(out_result[i])
        del out_result[0]

        tar_result = {0 : target}
        for i in range(5) :
            func = getattr(self, 'enc_%d' %(i + 1))
            tar_result[i + 1] = func(tar_result[i])
        del tar_result[0]

        
        fpn = FeaturePyramidNetwork(
            in_channels_list=[out_result[f].shape[1] for f in out_result.keys()],
            out_channels=256,
        ).to('cuda')
        output_fp = fpn(out_result)
        target_fp = fpn(tar_result)
        loss = 0
        for i in range(1, 6) :
            loss += self.mse(output_fp[i], target_fp[i])
        return loss


    def cal_target_dir(self, img) :
        
        feat = self.enc_8(img)
        # with torch.no_grad() :
        #     feat_norm = feat.clone() / feat.clone().norm(dim=-1, keepdim=True)
        # return feat_norm
        return feat
        
    def dir_loss(self, output, ref, target) :
        # output_feat = self.enc_8(output)
        # ref_feat = self.enc_8(ref)
        output_feat = self.cal_target_dir(output)
        ref_feat = self.cal_target_dir(ref)
        dir = output_feat - ref_feat

        if dir.sum() == 0 :
            output_feat = self.cal_target_dir(output + 1e-6)
            dir = output_feat - ref_feat
        # return self.mse(output_feat, ref_feat)
        return (1 - torch.cosine_similarity(dir, target)).mean()

    def gram_mat(self, features) :
        b, c, h, w = features.size()
        features = features.view(b*c, h*w)
        gram = torch.matmul(features, features.t())
        return gram.div(b*c*h*w)
        
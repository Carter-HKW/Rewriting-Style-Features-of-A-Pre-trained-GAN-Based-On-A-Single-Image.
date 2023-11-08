import argparse
import torch

from torchvision import transforms


from option.test_option import TestOptions
from model.reference_model import Reference_model
from util.visualizer import Visualizer
import os

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_G', type=str)
    parser.add_argument('--archG', default = 'stylegan3')
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    opt = parser.parse_args()
    device = 'cuda' 
    #pretrained model
    model = Reference_model(opt)
    model.netG.load_state_dict(torch.load('./best_record/horse/sketch/best_model.pth'))
    model.netG.load_state_dict(torch.load('./rewrite/weight/sg3r_horse/horse_camel_back.pth',map_location=device), strict=False)
    # geometric = torch.load('./rewrite/weight/sg3r_house/house_curly_roof.pth')
    # print(geometric)
    # model.netG.eval()
    #original model
    ref_m = Reference_model(opt)
    ref_m.netG.load_state_dict(torch.load('./rewrite/weight/sg3r_horse/horse_camel_back.pth',map_location=device), strict=False)

    cs = Reference_model(opt)
    cs.netG.load_state_dict(torch.load('./best_record/horse/sketch/best_model.pth'))
    Visual = Visualizer(opt)
    
    
    for i in range(10) :
        z = torch.randn(1, model.netG.z_dim, device='cuda')
        w_mix = model.netG.mapping(z, None, truncation_psi=0.7)
        save(i, w_mix)
        output = model(w_mix)
        ref = ref_m(w_mix)
        style = cs(w_mix)
        visual = {"output" : output, 'ref' : ref, 'style' : style}
        Visual.display_current_result(i, visual)
    
        

def save(num, w) :
    save_path = os.path.join('./best_record/ablation/latent', '%d_w.pth' %num)
    torch.save(w, save_path)
if __name__ == '__main__' :
    main()
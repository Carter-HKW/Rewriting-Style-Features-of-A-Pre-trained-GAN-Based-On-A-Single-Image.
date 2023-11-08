import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lpips
Lpips = lpips.LPIPS(net='alex').to('cuda')
mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
from option.test_option import TestOptions
from model.reference_model import Reference_model
from datasets.dataset import GanDataset
from transfer.get_transfer import CAST
from model.networks.net import vgg_loss
from util.visualizer import Visualizer
import os

def main() :
    opt = TestOptions().parse()
    device = 'cuda' if opt.gpus > 0 else 'cpu'
    #pretrained model
    model = Reference_model(opt)
    model.netG.load_state_dict(torch.load(opt.result_G))
    model.netG.eval()
    #original model
    ref_model = Reference_model(opt)

    # transfer = CAST()
    # transfer.setup()
    # test_data = GanDataset(opt, phase='test')
    # dataloader = DataLoader(test_data, batch_size = 1)
    Visual = Visualizer(opt)
    
    loss = 0
    total = 0
    # latent = torch.load(os.path.join('./best_record/ablation/latent/house', '9_w.pth'))
    # w = latent.to(device)
    # output = model(w)
    # visual = {'output' : output}
    # Visual.display_current_result(0, visual)
    for i in range(10) :
        z = torch.randn(1, model.netG.z_dim, device='cuda')
        w_mix = model.netG.mapping(z, None, truncation_psi=0.7)
        save(i, w_mix)
        output = model(w_mix)
        ref = ref_model(w_mix)
        visual = {"output" : output, 'ref' : ref}
        Visual.display_current_result(i, visual)
    # for name, param in model.named_parameters():
    #     print(name)
    # for name, param in model.named_modules():
    #     print(name)
    # for step, data in enumerate(dataloader) :
    #     with torch.no_grad() :
    #         latents = data['latents'].to(device)
    #         output = model(latents)
    #         # target = ref_model(latents)
    #         # target, style = transfer.forward(target)
    #         # target = target.to(device)
    #         # style = style.to(device)
    #         # target = transforms.Resize(512)(target)
    #         # style = transforms.Resize(512)(style)
    #         target = data['target'].to(device)
    #         target = transforms.Resize(512)(target)
    #         loss += eval(output, target, opt)

    #         #save output
    #         visual = {"output" : output, 'target' : target}
    #         Visual.display_current_result(step, visual)
    #         total += 1

    # save eval result
    # loss /= total
    # Visual.save_test_result(loss)

        
def eval(output, target, opt) :
    #device = 'cuda' if opt.gpus > 0 else 'cpu'
    
    loss_style = vgg_loss(opt)
    return loss_style.get_S_loss(output, target)
#  + loss_style.get_S_loss(output, target) + loss_style.get_C_loss(output, target) torch.mean(Lpips(output, target))
def save(num, w) :
    save_path = os.path.join('./best_record/ablation/latent', '%d_w.pth' %num)
    torch.save(w, save_path)
if __name__ == '__main__' :
    main()
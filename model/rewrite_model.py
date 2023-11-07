import torch
import model.networks as network
from util.util import slice_ordered_dict
from lib.dissect.param_tool import merge_state_dict
from lib.dissect.param_tool import get_params_from_module
import os
from collections import OrderedDict

class RewriteModel(torch.nn.Module) :
    """
    Finetune model
    """
    def __init__(self, opt) :
        super().__init__()
        self.opt = opt
        self.device = 'cuda' if self.opt.gpus > 0 else 'cpu'
        self.netG = network.define_G(opt).to(self.device)

        #layer
        module_dict = network.get_modules(opt.archG, self.netG, mode=opt.finetune_mode)
        # if opt.update_layer == 'all' :
        #     start = 0
        #     end = len(module_dict) - 1
        # elif '-' in opt.update_layer :
        #     start, end = [int(s) for s in opt.update_layer.split('-')]
        # else :
        #     start = int(opt.update_layer)
        #     end = start
        
        # self.update_layers = slice_ordered_dict(module_dict, start, end + 1)
        self.update_layers = OrderedDict(list(module_dict.items())[6:14] )
        

        # if opt.is_Train :
        #     self.update_params = self.set_update_params()
    
            
            
    def forward(self, w_latents) :
        w_latents = w_latents.to(self.device)
        return self.netG.synthesis(w_latents,force_fp32=True)
    
    def get_update(self) :
        param = []
        for name, mod in self.update_layers.items() :
            p = get_params_from_module(mod, prefix=name, exclude_children=True)

            #only update weight or both weight and bias
            if self.opt.weight_only :
                weight_name = f'{name}.weight'
                assert weight_name in p.keys(), f'{weight_name} should be in the module'
                param.append({weight_name : p[weight_name]})
            else :
                param.append(p)
        d = merge_state_dict(*param)
        return list(d.values())
    def get_all_param(self) :
        return list(self.netG.parameters())
    

    def save(self, epoch, best_valid=False) :
        if best_valid :
            save_path = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'best_model.pth')
        else :
            save_path = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'epoch_%d_model.pth' %epoch)
        torch.save(self.netG.state_dict(), save_path)


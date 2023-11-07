import torch
import model.networks as network
class Reference_model(torch.nn.Module) :
    """
    reference
    """
    def __init__(self, opt) :
        super().__init__()
        self.opt = opt
        self.device = 'cuda' if self.opt.gpus > 0 else 'cpu'
        #self.device = 'cpu'
        self.netG = network.define_G(opt).to(self.device)
    
    def forward(self, w_latents) :
        w_latents = w_latents.to(self.device)
        return self.netG.synthesis(w_latents,force_fp32=True)
    def get_all_param(self) :
        return list(self.netG.parameters())
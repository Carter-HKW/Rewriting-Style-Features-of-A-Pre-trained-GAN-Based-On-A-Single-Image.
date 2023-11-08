import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import numpy as np
import lpips
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import time
from PIL import Image
from os.path import join
from model.networks import set_requires_grad
from model.rewrite_model import RewriteModel
from model.reference_model import Reference_model
from datasets.dataset import GanDataset
from transfer.get_transfer import CAST
from transfer.IEST.use import IEST
from util.visualizer import Visualizer
from model.networks.net import vgg_loss
from model.networks.clip import ClipLoss
from model.networks.gan import Discriminator
import model.networks as network

class StyleTrainer() :
    def __init__(self, opt) :
        torch.backends.cudnn.benchmark = opt.cudnn_benchmark
        self.opt = opt
        #model
        self.model = RewriteModel(opt)
        self.ref = Reference_model(opt)
        #data
        # self.data = GanDataset(opt, phase='train')
        # self.dataloader = DataLoader(self.data, batch_size = opt.batch_size)
        #transfer
        self.transfer = CAST()
        self.transfer.setup()
        self.test_transfer = IEST()
        #train
        self.train_param = self.model.get_update()
        self.lr = opt.lr
        self.d = Discriminator(input_channels=3, num_filters=64).to('cuda')
        # self.d = network.define_D(opt).to('cuda')
        # self.d.train().requires_grad_(False)
        # self.d.c_dim = 0
        self.adver_loss = torch.nn.BCELoss()
        # self.adver_loss = torch.nn.BCEWithLogitsLoss()
        #self.optimizer = torch.optim.Adam(self.train_param, lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer = torch.optim.AdamW(self.train_param, lr = opt.lr, weight_decay=0.01)
        self.d_optimizer = torch.optim.AdamW(self.d.parameters(), lr = opt.lr / 2500 , weight_decay=0.01)

        self.crop = transforms.Compose([
            transforms.RandomCrop(64)
        ])

        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        self.lpips = lpips.LPIPS(net='alex').to(self.model.device)
        self.lpips_train = lpips.LPIPS(net='vgg').to(self.model.device)
        self.loss_style = vgg_loss(opt)
        
        
        if self.opt.lr_schedule == 'cosine' :
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        elif self.opt.lr_schedule == 'step' :
            self.scheduler = StepLR(self.optimizer, step_size=200, gamma=0.5)
            self.scheduler_d = StepLR(self.d_optimizer, step_size=200, gamma=0.5)
        else :
            self.scheduler = None
        
        #valid
        ###save best loss
        self.best_loss = 100000
        self.valid_data = GanDataset(self.opt, phase='valid')
        self.valid_dataloader = DataLoader(self.valid_data, batch_size = 5)
        for step, data in enumerate(self.valid_dataloader) :
            self.valid_target = data['target'].to(self.model.device)
        #visual
        self.visual = Visualizer(opt)
        
    
    def fit(self) :
        start_time = time.time()
        set_requires_grad(self.ref.get_all_param(), False)
        set_requires_grad(self.model.get_all_param(), False)
        set_requires_grad(self.train_param, True)
        # set_requires_grad(list(self.d.parameters()), True)
        
        if self.opt.train_continue is not None :
            self.model.netG.load_state_dict(torch.load(self.opt.train_continue))
        total_iter = 0

        
        for epoch in range(self.opt.epoch):
            train_loss = 0
            for step in range(2) : 
                torch.autograd.set_detect_anomaly(True)

                total_iter += 1
                if self.opt.lr_schedule == 'karras' :
                    self.adjust_lr(total_iter)
                
                
                
                #data augmentation
                if not self.opt.no_data_aug :
                    with torch.no_grad() :
                        z = torch.randn(self.opt.batch_size, self.model.netG.z_dim, device=self.model.device)
                        latents = self.model.netG.mapping(z, None, truncation_psi=0.7)
                        

                output = self.model(latents)
                output = transforms.Resize(256)(output)
                
                
                #transform
                with torch.no_grad() :
                    ref_output = self.ref(latents)
                    #target = self.test_transfer.forward(ref_output)
                    target, style= self.transfer.forward(ref_output)
                    target = target.to('cuda')
                    #ran_target = transforms.Resize(256)(target)
                    # style = style.to('cuda')
                    # style = transforms.Resize(256)(style)
                    # ref_output = transforms.Resize(256)(ref_output)
                
                

                
                #gan loss
                
                self.d_optimizer.zero_grad()
                # real_labels = torch.ones(latents.shape[0], 1).to(self.model.device)
                # fake_labels = torch.zeros(latents.shape[0], 1).to(self.model.device)
                real_pred = self.d(target, target)
                fake_pred = self.d(output.detach(), target)
                
                real_labels = torch.ones_like(real_pred).to(self.model.device)
                fake_labels = torch.zeros_like(fake_pred).to(self.model.device)
                real_d_loss = self.adver_loss(real_pred, real_labels)
                
                fake_d_loss = self.adver_loss(fake_pred, fake_labels)
                
                d_loss = (real_d_loss + fake_d_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()

                #loss
                loss = 0
                losses = {}
                
                if self.opt.loss_l1 :
                    losses['L1'] = self.l1(output, target)
                if self.opt.loss_mse :
                    losses['MSE'] = self.mse(output, target)
                if self.opt.loss_lpips :
                    losses['LPIPS'] = torch.mean(self.lpips_train(output, target))
                if self.opt.loss_style :
                    losses['Style'] = self.loss_style.get_S_loss(output, target)
                    
                
                if self.opt.loss_content :
                    losses['Content'] = self.loss_style.get_C_loss(output, target)
                if self.opt.loss_fpn :
                    losses['FPN'] = self.loss_style.get_fpn_loss(output, target)
                if self.opt.loss_random :
                    losses['random'] = self.random_loss(output, style)
                if self.opt.loss_dir :
                    losses['dir'] = self.train_clip.cal_dir_loss(output, ref_output)

                output_pred = self.d(output, target)
                
                losses['GAN'] = self.adver_loss(output_pred, real_labels)
                losses['DIS'] = d_loss
                loss +=  0.5*losses['GAN'] + 0.001*self.TV_loss(output) + 1.5 * losses['LPIPS'] + losses['Style'] + 1.5 * losses['Content']
                # 0.5*losses['GAN'] + 0.001*self.TV_loss(output)  + 1.5 * losses['LPIPS'] + losses['Style'] + 1.5 * losses['Content']
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                

                
                train_loss += (loss - 0.001 * self.TV_loss(output)) * latents.shape[0]
                # train_loss += loss * latents.shape[0]

                self.visual.print_current_loss(epoch, step, losses)
                if self.opt.display :
                    info = losses.copy()
                    if self.scheduler is not None :
                        info['lr'] = self.scheduler.get_last_lr()[0]
                    else :
                        info['lr'] = self.optimizer.param_groups[0]['lr']
                    self.visual.plot_current_log(total_iter, info)
                
                if epoch % 100 == 0 or epoch == self.opt.epoch - 1 :
                    visual = {'output' : output, 'ref' : ref_output, 'target' : target}

                    self.visual.display_current_result(total_iter, visual)

            #validation
            train_loss /= 2 * self.opt.batch_size
            val_loss, valid_img, check_overfit = self.do_valid()

            if epoch % 25 == 0 or epoch == self.opt.epoch - 1 :
                self.visual.display_current_result(epoch, valid_img)

            if val_loss < self.best_loss :
                self.best_loss = val_loss
                #self.visual.record_best(val_loss)
                self.model.save(epoch, best_valid=True)

            self.visual.show_valid_loss(epoch, {"Train" : train_loss, "Valid" : val_loss, "Check" : check_overfit})

            
            
            if self.scheduler is not None :
                self.scheduler.step()
                self.scheduler_d.step()
            #save model
            if epoch == self.opt.epoch - 1:
                self.model.save(epoch)
        # for epoch in range(self.opt.epoch):
        #     train_loss = 0
        #     for step, d in enumerate(self.dataloader): 
        #         torch.autograd.set_detect_anomaly(True)

        #         total_iter += 1
        #         if self.opt.lr_schedule == 'karras' :
        #             self.adjust_lr(total_iter)

        #         latents = d['latents'].to(self.model.device)
                
        #         #data augmentation
        #         if not self.opt.no_data_aug :
        #             with torch.no_grad() :
        #                 z = torch.randn(latents.shape[0], self.model.netG.z_dim, device=self.model.device)
        #                 w_mix = self.model.netG.mapping(z, None, truncation_psi=0.7)
        #                 m_start, m_end = [int(s) for s in self.opt.stylemix_layers.split(',')]
        #                 latents[:, m_start:m_end, :] = w_mix[:, m_start:m_end, :]

        #         output = self.model(latents)
        #         #output = transforms.Resize(256)(output)
                
                
        #         #transform
        #         with torch.no_grad() :
        #             ref_output = self.ref(latents)
                    
        #             target, style= self.transfer.forward(ref_output)
        #             target = target.to('cuda')
        #             #ran_target = transforms.Resize(256)(target)
        #             #style = style.to('cuda')
        #             #style = transforms.Resize(256)(style)
        #             #ref_output = transforms.Resize(256)(ref_output)
                
                

                
        #         #gan loss
                
        #         self.d_optimizer.zero_grad()
        #         real_labels = torch.ones(latents.shape[0], 1).to(self.model.device)
        #         fake_labels = torch.zeros(latents.shape[0], 1).to(self.model.device)
        #         real_pred = self.d(target)
        #         fake_pred = self.d(output.detach())
                
        #         real_d_loss = self.adver_loss(real_pred, real_labels)
                
        #         fake_d_loss = self.adver_loss(fake_pred, fake_labels)
                
        #         d_loss = real_d_loss + fake_d_loss
        #         d_loss.backward()
        #         self.d_optimizer.step()

        #         #loss
        #         loss = 0
        #         losses = {}
                
        #         if self.opt.loss_l1 :
        #             losses['L1'] = self.l1(output, target)
        #         if self.opt.loss_mse :
        #             losses['MSE'] = self.mse(output, target)
        #         if self.opt.loss_lpips :
        #             losses['LPIPS'] = torch.mean(self.lpips(output, target))
        #         if self.opt.loss_style :
        #             losses['Style'] = self.loss_style.get_S_loss(output, target)
                    
                
        #         if self.opt.loss_content :
        #             losses['Content'] = self.loss_style.get_C_loss(output, target)
        #         if self.opt.loss_fpn :
        #             losses['FPN'] = self.loss_style.get_fpn_loss(output, target)
        #         if self.opt.loss_random :
        #             losses['random'] = self.random_loss(output, style)
        #         if self.opt.loss_dir :
        #             losses['dir'] = self.train_clip.cal_dir_loss(output, ref_output)

        #         output_pred = self.d(output, c=None)
                
        #         losses['GAN'] = self.adver_loss(output_pred, real_labels)
        #         losses['DIS'] = d_loss
        #         loss += losses['GAN'] + losses['Style'] + 0.01 * self.TV_loss(output) + 0.7*losses['Content']
        #         #  + losses['Style'] + 2 * losses['Content']  0.001 * self.TV_loss(output) + losses['LPIPS'] + losses['L1'] 
        #         self.optimizer.zero_grad(set_to_none=True)
        #         loss.backward()
        #         self.optimizer.step()

                

                
        #         train_loss += (loss - 0.01 * self.TV_loss(output)) * latents.shape[0]
        #         # train_loss += loss * latents.shape[0]

        #         self.visual.print_current_loss(epoch, step, losses)
        #         if self.opt.display :
        #             info = losses.copy()
        #             if self.scheduler is not None :
        #                 info['lr'] = self.scheduler.get_last_lr()[0]
        #             else :
        #                 info['lr'] = self.optimizer.param_groups[0]['lr']
        #             self.visual.plot_current_log(total_iter, info)
                
        #         if epoch % 100 == 0 or epoch == self.opt.epoch - 1 :
        #             visual = {'output' : output, 'ref' : ref_output, 'target' : target}

        #             self.visual.display_current_result(total_iter, visual)

        #     #validation
        #     train_loss /= 10
        #     val_loss, valid_img, check_overfit = self.do_valid()

        #     if epoch % 25 == 0 or epoch == self.opt.epoch - 1 :
        #         self.visual.display_current_result(epoch, valid_img)

        #     if val_loss < self.best_loss :
        #         self.best_loss = val_loss
        #         #self.visual.record_best(val_loss)
        #         self.model.save(epoch, best_valid=True)

        #     self.visual.show_valid_loss(epoch, {"Train" : train_loss, "Valid" : val_loss, "Check" : check_overfit})

            
            
        #     if self.scheduler is not None :
        #         self.scheduler.step()
        #     #save model
        #     if epoch == self.opt.epoch - 1:
        #         self.model.save(epoch)
        train_time = time.time() - start_time
        # self.model.save(self.opt.epoch, best_valid=True)
        self.visual.record_best(self.best_loss)
        print(train_time)
        print(self.best_loss)

    def adjust_lr(self, step) :
        
        if self.opt.lr_schedule == 'karras':
            # Apply styleGAN Learning rate schedule.
            initial_learning_rate = self.opt.lr
            lr_rampup_length = self.opt.lr_rampup_length
            lr_rampdown_length = self.opt.lr_rampdown_length
            total_steps = self.opt.epoch * math.ceil(10 / self.opt.batch_size)

            t = step / total_steps
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
            
    def do_valid(self) :
        
        val = 0
        v_loss = 0
        valid_img = {}
        for step, data in enumerate(self.valid_dataloader) :
            
            with torch.no_grad() :
                latents = data['latents'].to(self.model.device)
                output = self.model(latents)
                #ref = self.ref(latents)
                output = transforms.Resize(256)(output)
                #ref = transforms.Resize(256)(ref)
                # ref_output = self.ref(latents)
                # target = self.test_transfer.forward(ref_output)
                # target = target.to('cuda')
                # style = style.to('cuda')
                # real_labels = torch.ones(output.shape[0], 1).to(self.model.device)
                pred = self.d(output, self.valid_target)
                real_labels = torch.ones_like(pred).to(self.model.device)
                loss = self.adver_loss(pred, real_labels)
                valid_img['valid%i' %step] = output
                

                #eval
                val += torch.mean(self.lpips(output, self.valid_target)) 
                #check 
                v_loss += loss 
                #  + self.loss_style.get_S_loss(output, self.valid_target) + self.loss_style.get_C_loss(output, self.valid_target) torch.mean(self.lpips(output, self.valid_target)) self.mse(output, self.valid_target)
                #target = transforms.Resize(256)(target) self.loss_style.dir_loss(output, ref, self.valid_dir)self.valid_clip.cal_dir_loss(output, ref)
               
        
        return val, valid_img, v_loss



    def TV_loss(self, x) :
        dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        return torch.norm(dx) + torch.norm(dy)
    
    def random_loss(self, output, style) :
        
        #style_c = transforms.Resize(64)(style)
        loss = 0
        for i in range(16) :
            output_c = self.crop(output)
            with torch.no_grad() :
                target, style= self.transfer.forward(output_c)
                
                target = transforms.Resize([64, 64])(target).to('cuda')
                #ran_target = transforms.Resize(256)(target)
                #style = style.to('cuda')
                #style = transforms.Resize(256)(style)
                #ref_output = transforms.Resize(256)(ref_output)
            loss += torch.mean(self.lpips_train(output_c, target))
            # l = self.loss_style.get_S_loss(output_c, style_c)
            # if l > 20 :
            #     loss += l
        loss /= 16
        return loss    
        

    def normalize(self, x) :
        x = x.float()
        nor_x = x * 2.0 / 255.0 - 1.0
        return nor_x
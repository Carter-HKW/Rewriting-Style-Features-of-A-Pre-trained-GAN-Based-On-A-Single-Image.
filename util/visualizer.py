import os
import time
from util import util
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class Visualizer() :
    def __init__(self, opt) :
        self.opt = opt
        self.name = opt.name
        self.save_dir = os.path.join(opt.checkpoint_dir, self.name)
        self.image_dir = os.path.join(self.save_dir, 'image')
        self.output_dir = os.path.join(self.save_dir, 'result')

        
        self.writer = SummaryWriter()

        if opt.isTrain :
            #output image
            util.mkdirs(self.image_dir)

            self.log_name = os.path.join(opt.checkpoint_dir, self.name, 'loss_log.txt')
            with open(self.log_name, 'w') as log_file :
                now = time.strftime('%c')
                log_file.write( '----------best training result (%s)-------------\n' %now)
                log_file.write('epoch : %d\n' %opt.epoch)
                log_file.write('lr_scheduler : %s (lr = %.3f)\n' %(opt.lr_schedule, opt.lr))
                log_file.write('update layer : %s\n' %opt.update_layer)
        else :
            util.mkdirs(self.output_dir)     

    def print_current_loss(self, epoch, iter, losses) :
        message = '(epoch : %d, iter : %d) ' %(epoch, iter)
        for k, v in losses.items():
            message += '%s : %.3f ' %(k, v.mean())
        print(message)
        
    
    def show_valid_loss(self, epoch, losses) :
        message = '----------(epoch : %d) ' %(epoch)
        for k, v in losses.items():
            message += '%s : %.3f ' %(k, v.mean())
        print(message)
        self.plot_current_log(epoch, losses)
    
    def record_best(self, loss) :
        with open(self.log_name, 'a') as log_file :
            log_file.write('best eval : %.3f' %loss)
        
        
    def save_test_result(self, loss) :
        result_name = os.path.join(self.save_dir, 'test_loss.txt')
        with open(result_name, 'w') as a :
            a.write('%.4f' %loss)

    def plot_current_log(self, iter, losses) :
        for k, v in losses.items() :
            if k == 'lr' :
                self.writer.add_scalar('%s/iter' %k, v, iter)
            else :
                self.writer.add_scalar('%s/iter' %k, v.mean().item(), iter)
        
    def display_current_result(self, step, visuals) :
        result = {}
        for k, v in visuals.items():
            if isinstance(v, np.ndarray) :
                continue
            tile = min(8, v.size(0)) if v.dim() == 4 else False
            t = util.tensor2im(v, tile=tile)
            result[k] = t
        
        if self.opt.isTrain :
            for label, img in result.items() :
                assert len(img.shape) == 3, "visualization should be be a single RGB image"
                img = util.clip_img(img)
                img_path = os.path.join(self.image_dir, 'iter:%d_%s.png' %(step, label))
                util.save_image(img, img_path)
        else :
            for label, img in result.items() :
                assert len(img.shape) == 3, "visualization should be be a single RGB image"
                #img = util.clip_img(img)
                img_path = os.path.join(self.output_dir, '%s_%d.png' %(label, step))
                util.save_image(img, img_path)
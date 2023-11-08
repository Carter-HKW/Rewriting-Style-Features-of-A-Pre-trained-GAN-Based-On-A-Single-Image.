import argparse

class BaseOptions() :
    

    def initialize(self, parser) :
        #experiment specifics
        parser.add_argument('--gpus', type = int, default = 1, help = '#GPUs to use, or 0 means CPU')
        parser.add_argument('--isTrain', action='store_true')
        parser.add_argument('--name', type=str, default='experiment')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')

        #input
        parser.add_argument('--batch_size', type = int, default = 2, help = 'input size')
        parser.add_argument('--dataroot', type = str)
        parser.add_argument('--valid_data', type=str)
        
        #network
        parser.add_argument('--archG', default = 'stylegan3')
        parser.add_argument('--pretrained_G', required = True)

        #misc
        parser.add_argument('--cudnn_benchmark', default = True, type = bool, help='set torch.backends.cudnn.benchmark to True or not')

        #training
        parser.add_argument('--epoch', default = 1000, type = int)
        parser.add_argument('--no_data_aug', action='store_true')
        parser.add_argument('--stylemix_layers', type = str, default = '0,16')
        parser.add_argument("--finetune_mode", default='conv', choices=['conv', 'affine', 'all'])
        parser.add_argument('--update_layer', default='0-14')
        #parser.add_argument('--update_layer2', default='1-2')
        parser.add_argument('--weight_only', action='store_true')
        parser.add_argument('--train_continue', default=None)
        #learning rate
        parser.add_argument('--lr', default=0.05, type=float)
        parser.add_argument('--lr_schedule', default='karras', choices=['karras', 'cosine', 'step', None])
        parser.add_argument("--lr_rampup_length", default=0.05, type=float)
        parser.add_argument("--lr_rampdown_length", default=0.25, type=float)
        parser.add_argument("--beta1", default=0.9, type=float)
        parser.add_argument("--beta2", default=0.99, type=float)
        #loss
        parser.add_argument('--loss_l1', action='store_true')
        parser.add_argument('--loss_mse', action='store_true')
        parser.add_argument('--loss_random', action='store_true')
        parser.add_argument('--loss_lpips', action='store_true')
        parser.add_argument('--loss_style', action='store_true')
        parser.add_argument('--loss_content', action='store_true')
        parser.add_argument('--loss_fpn', action='store_true')
        parser.add_argument('--loss_dir', action='store_true')
        parser.add_argument('--vgg', default='model/networks/vgg_normalised.pth')

        #result
        parser.add_argument('--display', action='store_true')

        return parser
    
    def parse(self) :
        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)
        opt, unknown = parser.parse_known_args()
        return opt

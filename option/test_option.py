import argparse

class TestOptions() :
    

    def initialize(self, parser) :
        #equipment
        parser.add_argument('--gpus', type = int, default = 1, help = '#GPUs to use, or 0 means CPU')
        #test data root
        parser.add_argument('--test_data', type=str)

        #model
        parser.add_argument('--archG', default = 'stylegan3')
        parser.add_argument('--pretrained_G', type=str, required=True)
        parser.add_argument('--result_G', type=str, required=True)

        #not train
        parser.add_argument('--isTrain', type=bool, default=False)
        parser.add_argument('--vgg', default='model/networks/vgg_normalised.pth')
        
        #output dir
        #parser.add_argument('--output_dir', type=str)
        parser.add_argument('--name', type=str, default='experiment')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
        return parser
    
    def parse(self) :
        parser = argparse.ArgumentParser()
        parser = self.initialize(parser)
        opt, unknown = parser.parse_known_args()
        return opt

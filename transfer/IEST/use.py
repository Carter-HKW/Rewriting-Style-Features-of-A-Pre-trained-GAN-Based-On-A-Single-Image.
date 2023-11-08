import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from transfer.IEST import net


def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(256))
    transform = transforms.Compose(transform_list)
    return transform
def test_transform2():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# parser = argparse.ArgumentParser()

# # Basic options
# parser.add_argument('--content', type=str, default = 'input/content/1.jpg',
#                     help='File path to the content image')
# parser.add_argument('--style', type=str, default = 'input/style/1.jpg',
#                     help='File path to the style image, or multiple style \
#                     images separated by commas if you want to do style \
#                     interpolation or spatial control')
# parser.add_argument('--steps', type=str, default = 1)
# parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')
# parser.add_argument('--decoder', type=str, default = 'model/decoder_iter_160000.pth')
# parser.add_argument('--transform', type=str, default = 'model/transformer_iter_160000.pth')

# # Additional options
# parser.add_argument('--save_ext', default = '.jpg',
#                     help='The extension name of the output image')
# parser.add_argument('--output', type=str, default = 'output',
#                     help='Directory to save the output image(s)')

# # Advanced options

# args = parser.parse_args()

class IEST() :
    def __init__(self) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if not os.path.exists('test'):
#     os.mkdir('test')

        self.decoder = net.decoder
        self.transform = net.Transform(in_planes = 512)
        vgg = net.vgg

        self.decoder.eval()
        self.transform.eval()
        vgg.eval()

        self.decoder.load_state_dict(torch.load('./transfer/IEST/model/decoder_iter_160000.pth'))
        self.transform.load_state_dict(torch.load('./transfer/IEST/model/transformer_iter_160000.pth'))
        vgg.load_state_dict(torch.load('./transfer/IEST/model/vgg_normalised.pth'))

        self.norm = nn.Sequential(*list(vgg.children())[:1])
        self.enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

        self.norm.to(self.device)
        self.enc_1.to(self.device)
        self.enc_2.to(self.device)
        self.enc_3.to(self.device)
        self.enc_4.to(self.device)
        self.enc_5.to(self.device)
        self.transform.to(self.device)
        self.decoder.to(self.device)


    def forward(self, source) :
        content_tf = test_transform()
        style_tf = test_transform2()
        style = style_tf(Image.open('./transfer/style_data/monet.jpg'))
        style = style.to(self.device).unsqueeze(0)
        result = []
        i = 0
        for img in source :

            #content = content_tf(Image.open('input/content/%i.png'%i))
            content = img
            
            content = content.to(self.device).unsqueeze(0)

            with torch.no_grad():

                for x in range(1):

                    
                    
                    Content4_1 = self.enc_4(self.enc_3(self.enc_2(self.enc_1(content))))
                    Content5_1 = self.enc_5(Content4_1)
                
                    Style4_1 = self.enc_4(self.enc_3(self.enc_2(self.enc_1(style))))
                    Style5_1 = self.enc_5(Style4_1)
                
                    content = self.decoder(self.transform(Content4_1, Style4_1, Content5_1, Style5_1))
                    content = content_tf(content)
                    content.clamp(0, 255)
                    
                result.append(content)
                
                output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                            './transfer/IEST/test', splitext(basename('input/content/%i.png' %i))[0],
                            splitext(basename('input/style/1.jpg'))[0], '.jpg'
                        )
                save_image(content, output_name)
                i += 1
        result = torch.cat(result, dim=0)
        #print(result.size())
        return result

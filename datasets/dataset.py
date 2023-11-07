from os.path import join
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class GanDataset(Dataset) :

    def __init__(self, opt, phase) :
        super().__init__()
        self.phase = phase
        self.transform = transforms.Compose([
            # transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if phase == 'train' :
            self.data, self.length = self.createData(opt.dataroot)
        elif phase == 'valid' :
            self.data, self.length = self.createData2(opt.valid_data)
        elif phase == 'test' :
            self.data, self.length = self.createData2(opt.test_data)

        

    def __getitem__(self, index) :
        data = {}
        data['latents'] = self.data['latents'][index]
        if self.phase != 'train' :
            data['target'] = self.data['target'][index]
        return data

    

    def __len__(self) :
        return self.length




    def createData(self, data_dir) :
        with open(join(data_dir, 'counter'), 'r') as f :
            t = int(f.read())
        data = {'latents' : []}
        latent_dir = join(data_dir, 'latents')
        for i in range(t) :
            latent = torch.load(join(latent_dir, f'{i}_w.pth'))
            data['latents'].append(latent[0])
        
        return data, t
    
    def createData2(self, data_dir) :
        with open(join(data_dir, 'counter'), 'r') as f :
            t = int(f.read())
        data = {'latents' : [], 'target' : []}
        latent_dir = join(data_dir, 'latents')
        target_dir = join(data_dir, 'targets')
        for i in range(t) :
            latent = torch.load(join(latent_dir, f'{i}_w.pth'))
            data['latents'].append(latent[0])

            target = Image.open(join(target_dir, f'{i}.png')).convert('RGB')
            target = self.transform(target)
            data['target'].append(target)
        
        return data, t
    

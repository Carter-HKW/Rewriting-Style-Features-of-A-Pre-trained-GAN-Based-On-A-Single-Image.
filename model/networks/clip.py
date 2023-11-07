import torch
import clip
from torchvision import transforms
imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
class ClipLoss() :
    def __init__(self, style, ref) :
        self.device = 'cuda'
        self.clip, self.clip_preprocess = clip.load('ViT-B/32', self.device, jit=False)
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                            self.clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                            self.clip_preprocess.transforms[4:])      
        
        self.target_dir = self.cal_target_dir(style, ref)
        self.text_dir = self.text_feat('photo', 'watercolor art')
    def cal_img_feat(self, img) :
        img = self.preprocess(img)
        return self.clip.encode_image(img)
    def cal_target_dir(self, target, sources) :
        with torch.no_grad() :
            # pre_target = torch.unsqueeze(self.clip_preprocess(target), dim=0).to('cuda')
            # target_feat = self.clip.encode_image(pre_target)
            # target_feat = torch.mean(target_feat, dim=0, keepdim=True)
            target_feat = []
            for img in target :
                pre_img = torch.unsqueeze(self.clip_preprocess(img), dim=0).to('cuda')
                img_feat = self.clip.encode_image(pre_img)
                img_feat /= torch.norm(img_feat, dim=-1, keepdim=True)
                target_feat.append(img_feat)
            target_feat = torch.cat(target_feat, axis=0)
            target_feat = torch.mean(target_feat, dim=0, keepdim=True)

            src_feat = []
            for img in sources :
                pre_img = torch.unsqueeze(self.clip_preprocess(img), dim=0).to('cuda')
                img_feat = self.clip.encode_image(pre_img)
                img_feat /= torch.norm(img_feat, dim=-1, keepdim=True)
                src_feat.append(img_feat)
            src_feat = torch.cat(src_feat, axis=0)
            src_feat = torch.mean(src_feat, dim=0, keepdim=True)

            dir = target_feat - src_feat
            dir /= torch.norm(dir, dim=-1, keepdim=True)
        return dir

    def cal_dir_loss(self, output, ref) :
        output_feat = self.cal_img_feat(output)
        ref_feat = self.cal_img_feat(ref)
        dir = output_feat - ref_feat
        if dir.sum() == 0 :
            output_feat = self.cal_img_feat(output + 1e-6)
            dir = output_feat - ref_feat
        dir /= dir.clone().norm(dim=-1, keepdim=True)
        return (1 - torch.nn.functional.cosine_similarity(dir, self.text_dir)).mean()
    
    def text_feat(self, source, target) :
        template_text = self.compose_text_with_templates(target, imagenet_templates)
        tokens = clip.tokenize(template_text).to('cuda')
        text_features = self.clip.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        template_source = self.compose_text_with_templates(source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to('cuda')
        text_source = self.clip.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        
        dir = (text_features - text_source).mean(axis=0, keepdim=True)
        dir /= dir.norm(dim=-1, keepdim=True)
        return dir
    
    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
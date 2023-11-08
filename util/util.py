from collections import OrderedDict
import numpy as np
import os
from PIL import Image

def slice_ordered_dict(d, start, end):
    assert type(d) == OrderedDict, f"d must be an OrderedDict, but get type {type(d)}"
    return OrderedDict(list(d.items())[start:end])

def tile_images(imgs, picturesPerRow=4) :
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


def tensor2im(img_tensor, imtype=np.uint8, tile=2) :
    if isinstance(img_tensor, list) :
        imgs = []
        for t in img_tensor :
            imgs.append(tensor2im(t))
        return imgs
    if len(img_tensor.shape) == 4 :
        imgs = []
        for t in img_tensor:
            single_img = tensor2im(t)
            imgs.append(single_img.reshape(1, *single_img.shape))
        imgs = np.concatenate(imgs, axis=0)
        if tile is not False :
            images_tiled = tile_images(imgs, picturesPerRow=tile)
            return images_tiled
        else:
            return imgs
        
    
    img = img_tensor.detach().cpu().float().numpy() if type(img_tensor) is not np.ndarray else img_tensor
    
    img = (np.transpose(img, (1, 2, 0)) + 1) * 0.5 * 255.0   #transpose and scaling
    img = np.clip(img, 0, 255)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img.astype(imtype)

def mkdirs(path) :
    if isinstance(path, list) and not isinstance(path, str) :
        for p in path :
            if not os.path.exists(p) :
                os.makedirs(p)
    else :
        if not os.path.exists(path) :
                os.makedirs(path)


def clip_img(img, size = 1024) :
    # h, w = img.shape[:2]
    # if max(h, w) <= size :
    #     return img
    # ratio = size / max(h, w)
    # new_h = int(h * ratio)
    # new_w = int(w * ratio)
    resized = Image.fromarray(img).resize((size, size), Image.LANCZOS)
    return np.asarray(resized)

def save_image(img, img_path, aspect_ratio = 1.0) :
    image_pil = Image.fromarray(img)
    h, w, _ = img.shape
    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(img_path)
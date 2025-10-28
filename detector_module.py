
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class OpenWorldDetector:
    def __init__(self, mask_generator, dino_model, dino_transform, device='cuda'):
        self.mask_generator = mask_generator
        self.dino = dino_model
        self.tf = dino_transform
        self.device = device

    def segment(self, image_pil):
        return self.mask_generator.generate(np.array(image_pil))

    def crop_by_mask(self, image_pil, mask_dict):
        x, y, w, h = map(int, mask_dict["bbox"])
        full = np.array(image_pil)
        seg = mask_dict["segmentation"].astype(bool)
        obj = full.copy()
        obj[~seg] = 0
        obj = obj[y:y+h, x:x+w]
        return Image.fromarray(obj)

    def featurize(self, crop_pil):
        with torch.no_grad():
            t = self.tf(crop_pil).unsqueeze(0).to(self.device)
            f = self.dino(t)
            f = F.normalize(f, dim=1)
        return f.squeeze(0).cpu().numpy()

    def match(self, target_crop, reference_crops):
        tf = self.featurize(target_crop)
        rfs = np.stack([self.featurize(rc) for rc in reference_crops])
        sims = tf @ rfs.T
        return sims


"""
---------------------------------------------------------------------
OpenWorldDetector Module
---------------------------------------------------------------------
Implements the detector stage of GRPoseNet Phase 2.
Uses SAM (Segment Anything) for zero-shot segmentation
and DINOv2 for semantic feature extraction and matching.
---------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class OpenWorldDetector:
    """
    Open-world object detector combining SAM for segmentation
    and DINOv2 for semantic feature extraction and cosine-based matching.
    """

    def __init__(self, mask_generator, dino_model, dino_transform, device='cuda'):
        # Initialize model components and device
        self.mask_generator = mask_generator     # SAM (Segment Anything) instance
        self.dino = dino_model                   # DINOv2 feature extractor
        self.tf = dino_transform                 # preprocessing transform for DINOv2
        self.device = device

    def segment(self, image_pil):
        """
        Perform zero-shot segmentation using SAM.
        Returns a list of mask dictionaries containing 'bbox' and 'segmentation'.
        """
        return self.mask_generator.generate(np.array(image_pil))

    def crop_by_mask(self, image_pil, mask_dict):
        """
        Crop an object region from the image defined by a given mask.
        Background pixels are removed for clean feature extraction.
        """
        x, y, w, h = map(int, mask_dict["bbox"])
        full = np.array(image_pil)
        seg = mask_dict["segmentation"].astype(bool)

        # retain object pixels only
        obj = full.copy()
        obj[~seg] = 0
        obj = obj[y:y+h, x:x+w]
        return Image.fromarray(obj)

    def featurize(self, crop_pil):
        """
        Extract a normalized DINOv2 feature vector for the given object crop.
        """
        with torch.no_grad():
            t = self.tf(crop_pil).unsqueeze(0).to(self.device)  # preprocessing
            f = self.dino(t)                                   # forward through DINOv2
            f = F.normalize(f, dim=1)                          # L2 normalization
        return f.squeeze(0).cpu().numpy()

    def match(self, target_crop, reference_crops):
        """
        Compute cosine similarity between a target crop and a set of reference crops.
        Returns an array of similarity scores.
        """
        tf = self.featurize(target_crop)
        rfs = np.stack([self.featurize(rc) for rc in reference_crops])
        sims = tf @ rfs.T  # dot product (since normalized, equivalent to cosine similarity)
        return sims


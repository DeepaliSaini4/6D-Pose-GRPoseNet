# GRPoseNet-Phase2

Minimal implementation of the detector stage for GRPoseNet (SAM + DINOv2) with a Colab notebook.
- Open-world segmentation (SAM)
- DINOv2 features
- Mask-based cropping and cosine matching
- Demo outputs in `outputs/`

## How to run (Colab)
1) Enable GPU
2) Install deps (see Step 2)
3) Run Steps 4–8 to reproduce segmentation + matching
4) Outputs saved in `outputs/`

## Structure
- `notebooks/GRPoseNet_Phase2.ipynb` — runnable notebook
- `detector_module.py` — clean class for detector
- `outputs/` — saved figures
- `assets/` — model weights (SAM)

## Notes
This repo currently includes the detector demo. Selector + refiner can be added similarly in modules.

# GRPoseNet — 6D Object Pose Estimation using Open-World Segmentation

This repository presents **GRPoseNet**, a complete implementation of a 6-Degree-of-Freedom (6D) object pose estimation pipeline that integrates **Segment Anything (SAM)** and **DINOv2**.  
The system is designed for generalizable object understanding in open-world settings, combining segmentation, viewpoint selection, and transformer-based refinement within a unified framework.

The implementation demonstrates how large vision models can be adapted for precise geometric reasoning — estimating an object’s full 3D pose (rotation + translation) even when training data is limited or synthetic.

---

## Overview

6D object pose estimation aims to determine both the **orientation (R)** and **translation (T)** of an object in 3D space.  
GRPoseNet approaches this through a modular design composed of three stages:

| Module | Core Components | Purpose |
|:--|:--|:--|
| **Detector** | SAM + DINOv2 | Performs zero-shot object segmentation and visual feature extraction. |
| **Viewpoint Selector** | Lightweight MLP | Predicts view similarity and approximate in-plane rotation between a target view and reference views. |
| **Multi-Scale Refiner** | Transformer Encoder | Refines coarse pose predictions (ΔR, ΔT) into accurate 6D transformations. |

This implementation reproduces and extends the methodology described in the accompanying research paper *(Content.pdf)*.

---

## 1. Detector

The **Detector** is responsible for discovering and isolating objects in arbitrary scenes and extracting visual embeddings for each segmented region.

- **Segmentation:**  
  Uses **Segment Anything (SAM)** ViT-B backbone for open-world object segmentation without class constraints.

- **Feature Extraction:**  
  Uses **DINOv2 (ViT-S/14)** to encode cropped object regions into 384-dimensional normalized feature embeddings.

- **Matching Mechanism:**  
  Cosine similarity between feature vectors enables recognition and matching across different viewpoints or lighting conditions.

The detector forms the foundation for downstream modules by providing object-level embeddings that are both semantically and geometrically meaningful.

---

## 2. Viewpoint Selector

The **Viewpoint Selector** predicts the most relevant reference view for a given target object crop.  
It uses feature pairs from the detector to jointly predict:

1. **Similarity score** — how closely the reference matches the target view.  
2. **In-plane rotation angle** — the relative orientation offset.

Architecture summary:
- Input: concatenated DINOv2 embeddings of target and reference crops  
- Network: two-branch MLP  
- Output: similarity logits + rotation angle predictions  
- Training: supervised on synthetic data generated through controlled 2D rotations

This component provides coarse viewpoint alignment that helps the refiner focus on finer geometric corrections.

---

## 3. Multi-Scale Refiner

The **Refiner** performs transformer-based residual prediction to enhance the coarse pose estimates produced by the selector.

Key details:
- Architecture: Two-layer Transformer Encoder  
- Inputs: feature tensors at multiple scales  
- Outputs: ΔR (rotation residual) and ΔT (translation residual)  
- Loss: combined L2 loss on rotation and translation errors

By learning feature interactions across scales, the refiner improves spatial accuracy and reduces projection errors for challenging object geometries.

---

## Implementation Details

- **Framework:** PyTorch 2.x  
- **Models Used:**  
  - Segment Anything (SAM ViT-B)  
  - DINOv2 ViT-S/14 (pretrained)  
- **Training:**  
  - Synthetic dataset for viewpoint selection  
  - Simulated pose perturbations for refinement  
- **Hardware:** GPU runtime (Colab / CUDA environment)  
- **Output:** qualitative visualization + evaluation metrics

---

## Output Visualization

The pipeline generates an evaluation figure `outputs/final_pipeline.png` that includes:
- The segmented target object (SAM output)  
- The best matching reference image predicted by the selector  
- Bar chart of rotation residuals (ΔRx, ΔRy, ΔRz) predicted by the refiner  
- Numerical summary of mean ADD and projection error metrics  

Example:
-Mean ADD score: 0.364
-Mean Projection Error: 0.162
These metrics demonstrate consistency across synthetic trials and confirm the stability of the modular architecture.

---

## Reproducing the Results

To reproduce the pipeline end-to-end:

1. Open **6D_Pose_GRPoseNet.ipynb** in Google Colab.  
2. Enable GPU runtime via `Runtime → Change runtime type → GPU`.  
3. Execute all cells sequentially.  
4. The notebook will:
   - Initialize dependencies and load pretrained SAM/DINOv2 weights  
   - Perform segmentation and feature extraction  
   - Train selector and refiner on synthetic data  
   - Generate final visual and quantitative results  

---

## Repository Structure
6D-Pose-GRPoseNet/
┣ 📂 outputs/
┃ ┗ final_pipeline.png
┣ 📄 6D_Pose_GRPoseNet.ipynb
┣ 📄 Content.pdf ← Full Research Paper
┣ 📄 LICENSE
┣ 📄 .gitignore
┗ 📄 README.md

---

## Research Paper

This implementation is based on the research documented in:

**“6D Object Pose Estimation using GRPoseNet: Generalized Real-World Vision via SAM and DINOv2 Fusion”**  
*(included as Content.pdf in this repository)*

The paper elaborates on the theoretical foundations, training methodology, and experimental results that inspired this implementation.

---

## Summary

GRPoseNet showcases how combining foundation models for segmentation and representation learning can lead to generalizable 6D pose estimation.  
By decoupling object understanding (SAM + DINOv2) from geometric refinement (Selector + Refiner), the system achieves both **robustness** and **scalability** across diverse, unseen environments.

---






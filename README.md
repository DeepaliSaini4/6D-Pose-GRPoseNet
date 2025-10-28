# GRPoseNet — 6D Object Pose Estimation using Open-World Segmentation

**GRPoseNet** is a complete 6-Degree-of-Freedom (6D) object pose estimation pipeline that integrates **Segment Anything (SAM)** and **DINOv2**.
It performs open-world segmentation, viewpoint similarity prediction, and pose refinement to achieve accurate 3D pose recovery for unseen objects.

This repository demonstrates how large-scale vision models can be adapted for geometric reasoning — estimating both the orientation and position of arbitrary objects without requiring class-specific training data.

---

## Overview

6D object pose estimation determines both the **rotation (R)** and **translation (T)** of an object in 3D space.
GRPoseNet follows a modular structure consisting of three core stages:

| Module              | Core Components     | Purpose                                                                              |
| ------------------- | ------------------- | ------------------------------------------------------------------------------------ |
| Detector            | SAM + DINOv2        | Performs zero-shot segmentation and extracts dense visual embeddings.                |
| Viewpoint Selector  | Lightweight MLP     | Predicts similarity and relative rotation between a target view and reference views. |
| Multi-Scale Refiner | Transformer Encoder | Refines coarse rotation and translation into precise 6D pose values.                 |

Each component is designed to function independently but integrates seamlessly into the end-to-end pipeline.

---

## 1. Detector

The Detector identifies and isolates objects in arbitrary images using SAM, then encodes them into feature representations with DINOv2.

* **Segmentation:** Segment Anything (ViT-B) produces binary object masks in open-world scenarios.
* **Feature Extraction:** DINOv2 (ViT-S/14) generates 384-dimensional feature embeddings for each cropped region.
* **Matching:** Cosine similarity between target and reference embeddings determines appearance consistency across views.

These embeddings serve as the input foundation for viewpoint selection and pose refinement.

---

## 2. Viewpoint Selector

The Viewpoint Selector predicts which reference image best matches a given target object view.
It also estimates the approximate in-plane rotation angle between the two.

Architecture Summary:

* Input: concatenated feature vectors of target and reference crops
* Network: dual-branch MLP for similarity and angle prediction
* Output: similarity logits and rotation angles
* Training: synthetic 2D rotations applied to base images

This module provides coarse alignment and prepares the data for geometric refinement.

---

## 3. Multi-Scale Refiner

The Refiner adjusts the predicted poses by learning residual corrections across multiple feature scales.

Key Details:

* Two-layer Transformer Encoder for cross-scale attention
* Outputs ΔR (rotation residual) and ΔT (translation residual)
* Optimized using combined L2 loss on both outputs
* Simulated synthetic data for training stability

The refiner ensures smoother orientation prediction and reduced projection error.

---

## Implementation Summary

* **Framework:** PyTorch, TorchVision
* **Pretrained Models:**

  * SAM ViT-B (for segmentation)
  * DINOv2 ViT-S/14 (for embeddings)
* **Training Data:** synthetic viewpoints and pose perturbations
* **Hardware:** optimized for GPU (Colab T4 / A100)
* **Output:** visual and numerical evaluation metrics

---

## Output Visualization

The pipeline produces the figure `outputs/final_pipeline.png`, showing:

* The segmented target object
* The most similar reference image predicted by the selector
* Bar graph of rotation residuals (ΔRx, ΔRy, ΔRz)
* Mean ADD score and projection error summary

Example:

```
Mean ADD score: 0.364
Mean Projection Error: 0.162
```

---

## How to Use

1. Open `6D_Pose_GRPoseNet.ipynb` in Google Colab.
2. Enable GPU: Runtime → Change runtime type → GPU.
3. Run all cells sequentially.
4. The notebook will automatically:

   * Load pretrained SAM and DINOv2 models
   * Perform segmentation and feature extraction
   * Train the Selector and Refiner
   * Generate final outputs and visualizations

---

## Repository Structure

```
6D-Pose-GRPoseNet/
 ├── outputs/
 │   └── final_pipeline.png
 ├── 6D_Pose_GRPoseNet.ipynb
 ├── Content.pdf            ← Full Research Paper
 ├── LICENSE
 ├── .gitignore
 └── README.md
```

---

## Research Paper

This implementation is based on the work titled:

**“6D Object Pose Estimation using GRPoseNet: Generalized Real-World Vision via SAM and DINOv2 Fusion”**
*(included as Content.pdf in this repository)*

The paper details the theoretical motivation, architectural design, and experimental validation behind this implementation.

---

## Summary

GRPoseNet demonstrates how integrating open-world segmentation (SAM) and self-supervised representation learning (DINOv2) can enable generalized 6D pose estimation.
By separating perception (Detector) and geometric reasoning (Selector + Refiner), it achieves adaptable, explainable, and transferable performance across unseen domains.

---






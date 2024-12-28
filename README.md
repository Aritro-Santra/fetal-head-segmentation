# Image Segmentation Pipeline

This project implements a modular pipeline for image segmentation with preprocessing, active contour-based segmentation, and evaluation using Intersection over Union (IoU). It is designed for ultrasound image segmentation but can be adapted for other imaging tasks.

## Features
- Manual or pre-defined Region of Interest (ROI) selection.
- Gamma correction and noise reduction with customizable filters.
- Active contour-based segmentation for precise boundary detection.
- IoU evaluation for segmentation quality.
- Modular and extendable design.

## Requirements
- Python 3.7+
- Required libraries:
  - OpenCV
  - NumPy
  - Matplotlib
  - Scikit-Image

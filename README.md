# Content-Aware Image Scaling (Seam Carving)

This project implements vertical seam carving to perform content-aware image scaling.  
It identifies and removes low-energy pixel seams to reduce image width while preserving 
important structures and edges.

## Features
- Computes pixel energy using Sobel gradient
- Uses dynamic programming to find minimum-energy seams
- Removes vertical seams to reduce image width
- Outputs a resized image with minimal distortion

## Requirements
- Python 3.10+
- numpy
- opencv-python

## Installation
Install dependencies:
pip install numpy opencv-python

## Usage
Place an input image named `input.jpg` in the project folder.

Run:
python content_aware_scaling.py

Outputs:
output.jpg

## Example Result
- Input size: 639 x 733  
- Output size: 634 x 733 (after removing 5 vertical seams)

## Files
- `content_aware_scaling.py` — main algorithm
- `input.jpg` — original test image
- `output.jpg` — content-aware scaled image
- `report.pdf` — final APA-formatted project writeup

## Author
Thiago Silva

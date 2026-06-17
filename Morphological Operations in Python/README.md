# Digital Image Processing – Morphological Operations in Python

## Overview

This project implements fundamental Morphological Image Processing operations in Python based on MATLAB examples from:

Rafael C. Gonzalez, Richard E. Woods — Digital Image Processing

The project is designed for:

- MSc Artificial Intelligence courses
- Digital Image Processing laboratories
- Google Colab execution
- MATLAB-to-Python translation practice

<img width="2565" height="2616" alt="textural_segmentation" src="https://github.com/user-attachments/assets/3abfdc73-e777-4b4e-8f89-54e1db75d1ce" />

---

# Implemented Operations

- Dilation
- Erosion
- Opening
- Closing
- Hit-or-Miss Transform

---

# Repository Structure

```text
.
├── morphology_combined_solution.ipynb
├── morphology_combined_solution.py
├── morphology_combined_assignment_report.docx
└── README.md
```

---

# Morphological Image Processing

Morphological processing analyzes image structures based on shapes.

Applications:

- Noise removal
- Object detection
- Segmentation preprocessing
- Shape analysis
- Boundary smoothing

These operations use a Structuring Element (Kernel).

Examples:

```python
np.ones((3,3))
np.ones((5,5))
disk(10)
```

---

# 1. Dilation

## Definition

Dilation expands white regions in a binary image.

Applications:

- Filling gaps
- Connecting nearby objects
- Enhancing object visibility

The project applies dilation using:

- 3×3 kernel
- 5×5 kernel
- 7×7 kernel
- 9×9 kernel

Three different random runs are used because random pixels are generated.

<img width="3580" height="804" alt="dilation_run_42" src="https://github.com/user-attachments/assets/0f399866-7f1c-4af2-936c-c8bf311302d5" />
<img width="3580" height="804" alt="dilation_run_7" src="https://github.com/user-attachments/assets/05715c67-b33d-428a-86ca-0a524ca7b4f1" />
<img width="3580" height="804" alt="dilation_run_2025" src="https://github.com/user-attachments/assets/04f82d7d-77cf-4109-9353-145d31a805ef" />

---

# 2. Erosion

## Definition

Erosion shrinks white regions.

Applications:

- Removing small noise
- Separating connected objects
- Boundary extraction

<img width="3580" height="804" alt="erosion_results" src="https://github.com/user-attachments/assets/5837146d-dc06-4edc-b700-f17481be4237" />

---

# 3. Opening

## Definition

Opening = Erosion + Dilation

Applications:

- Noise removal
- Removing small bright objects
- Cleaning binary images

Experiments:

- blobs image
- snowflakes image

<img width="2379" height="1890" alt="opening_results" src="https://github.com/user-attachments/assets/5d7f1398-a5fc-4d7c-bb1c-51300aef2f20" />

---

# 4. Closing

## Definition

Closing = Dilation + Erosion

Applications:

- Filling holes
- Connecting nearby regions
- Smoothing boundaries

The circles image is processed using a disk structuring element.

<img width="1953" height="1010" alt="closing_results" src="https://github.com/user-attachments/assets/76f5458a-d28a-40ae-bf63-57f85d33438d" />

---

# 5. Hit-or-Miss Transform

## Definition

Detects specific binary patterns.

Applications:

- Pattern recognition
- Template matching
- Skeleton analysis

<img width="1676" height="778" alt="hitmiss_results" src="https://github.com/user-attachments/assets/818bc7a2-4a9f-457d-9880-9b3c21dc09d4" />

---

# Libraries Used

| Library      | Purpose                  |
| ------------ | ------------------------ |
| NumPy        | Matrix operations        |
| OpenCV       | Morphological processing |
| Scikit-image | Structuring elements     |
| Matplotlib   | Visualization            |

---

# Installation

## Google Colab

```python
!pip install opencv-python scikit-image matplotlib scipy imageio
```

---

# Running

## Notebook

Open:

```text
morphology_combined_solution.ipynb
```

Run all cells.

## Python Script

```bash
python morphology_combined_solution.py
```

---

# Automatic Image Download

The notebook automatically downloads:

- cameraman.tif
- circles.png
- blobs.png
- snowflakes.png

No manual upload is required.

---

# MATLAB to Python Equivalents

| MATLAB    | Python                         |
| --------- | ------------------------------ |
| imdilate  | cv2.dilate                     |
| imerode   | cv2.erode                      |
| imopen    | cv2.morphologyEx               |
| imclose   | cv2.morphologyEx               |
| bwhitmiss | skimage.morphology.hit_or_miss |

---

# Educational Goals

This project helps students understand:

- Binary morphology
- Structuring elements
- Noise removal
- Shape processing
- Object connectivity
- Pattern matching

---

# References

1. Rafael C. Gonzalez, Richard E. Woods — Digital Image Processing
2. OpenCV Documentation
3. Scikit-image Documentation

---

# Author

M-A-Kiani
Prepared for:
MSc Artificial Intelligence — Digital Image Processing Course

# Digital Image Processing – Morphological Operations in Python

## Overview

This project implements fundamental Morphological Image Processing operations in Python based on MATLAB examples from:

Rafael C. Gonzalez, Richard E. Woods — Digital Image Processing

The project is designed for:

- MSc Artificial Intelligence courses
- Digital Image Processing laboratories
- Google Colab execution
- MATLAB-to-Python translation practice

<img width="5184" height="3456" alt="snowflakes" src="https://github.com/user-attachments/assets/a4276368-c109-474d-9cc6-cb740502d342" />
<img width="256" height="256" alt="circles" src="https://github.com/user-attachments/assets/efd7a2f5-58e5-4c1b-ae13-85217943154c" />
![cameraman](https://github.com/user-attachments/assets/9dac2442-69ae-4da9-9bd8-8df103f036f2)
<img width="256" height="254" alt="blobs" src="https://github.com/user-attachments/assets/45aa801e-9b53-4dcc-9fde-1aa00ac7778c" />

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

---

# 2. Erosion

## Definition

Erosion shrinks white regions.

Applications:

- Removing small noise
- Separating connected objects
- Boundary extraction

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

---

# 4. Closing

## Definition

Closing = Dilation + Erosion

Applications:

- Filling holes
- Connecting nearby regions
- Smoothing boundaries

The circles image is processed using a disk structuring element.

---

# 5. Hit-or-Miss Transform

## Definition

Detects specific binary patterns.

Applications:

- Pattern recognition
- Template matching
- Skeleton analysis

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

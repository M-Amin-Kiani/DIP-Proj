# 🎯 Open Vocabulary Video Tracking using SAM3

> Final Project for Digital Image Processing (M.Sc. Artificial Intelligence)

---

## Overview

This project demonstrates Open Vocabulary Video Object Tracking using Meta's Segment Anything Model 3 (SAM3).

Unlike traditional object tracking algorithms, SAM3 can track arbitrary objects using only a natural language prompt.

Example:

```
person
```

or

```
electric guitar
```

or

```
microphone
```

The model automatically:

- detects the target
- segments it
- tracks it
- computes the centroid
- exports trajectory
- generates visualization

---

# Outputs

After execution the following files are generated.

```
output.mp4
output.gif
trajectory.csv
trajectory.png
statistics.txt
```

---

# Pipeline

```
Video

↓

Text Prompt

↓

SAM3

↓

Segmentation Mask

↓

Bounding Box

↓

Centroid

↓

Trajectory

↓

Output Video
```

---

# Features

✅ Open Vocabulary Tracking

✅ Text Prompt

✅ Video Segmentation

✅ Centroid Extraction

✅ CSV Export

✅ Trajectory Visualization

✅ MP4 Rendering

✅ GIF Rendering

---

# Repository Structure

```
input/

outputs/

checkpoints/

src/

notebook/
```

---

# Installation

Clone repository

```bash
git clone https://github.com/USERNAME/SAM3-OpenVocabulary-Tracking.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download checkpoint

```
ModelScope
```

or

```
HuggingFace
```

Run notebook

```
SAM3_OpenVocabulary_Tracking.ipynb
```

---

# Example

Input

```
5-second concert video
```

Prompt

```
person
```

Outputs

```
output.mp4

trajectory.csv

trajectory.png

output.gif
```

---

# Centroid Computation

Centroid is computed using image moments

\[
c_x=\frac{m_{10}}{m_{00}}
\]

\[
c_y=\frac{m_{01}}{m_{00}}
\]

If the segmented area becomes too small, the center of the bounding box is used as a fallback.

---

# Statistics

The notebook reports

- tracked frames

- tracking success rate

- total trajectory length

- average object speed

---

# Results

The generated output video contains

- segmentation mask

- object boundary

- centroid

- bounding box

- trajectory

- frame number

- confidence score

---

# Technologies

Python

PyTorch

OpenCV

NumPy

Pandas

Matplotlib

Google Colab

SAM3

---

# References

Meta AI — Segment Anything Model 3

OpenCV Documentation

PyTorch Documentation

---

# Author

Mohammad Amin Kiani

M.Sc. Artificial Intelligence

University of Isfahan

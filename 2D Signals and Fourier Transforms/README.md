# 2D Signals and Fourier Transforms (Digital Image Processing)

## 📌 Project Overview
<img width="1356" height="2995" alt="image" src="https://github.com/user-attachments/assets/2262f635-99f9-452d-8ab7-aedb30e69910" />

This project explores fundamental **2D signals** and their corresponding **Fourier Transforms**, based on concepts from _Digital Image Processing (Gonzalez & Woods)_.

The goal is to visualize and understand the relationship between:

- **Spatial Domain (PSF - Point Spread Function)**
- **Frequency Domain (OTF - Optical Transfer Function)**

Both **2D (image)** and **3D surface visualizations** are used to provide deeper intuition.

---

## 🎯 Objectives

- Generate common 2D signals:
  - Rectangle
  - Pyramid
  - Gaussian
  - Peak (1/r)
  - Exponential Decay

- Compute their **2D Fourier Transform**
- Visualize results in:
  - Spatial Domain
  - Frequency Domain
  - 3D Surface Plots

- Analyze how spatial features affect frequency content

---

## 🧠 Key Concepts

### Spatial vs Frequency Domain

| Domain    | Meaning                         |
| --------- | ------------------------------- |
| Spatial   | Original image / signal         |
| Frequency | Rate of change (details, edges) |

---

### Important Observations

- Sharp edges → High frequency components
- Smooth signals → Low frequency concentration
- Gaussian remains Gaussian in frequency domain
- Rectangle produces sinc pattern
- Pyramid produces smoother (sinc²-like) response

---

## 🧪 Signals Implemented

### 1. Rectangle

- Sharp edges
- Produces **sinc pattern** in frequency

### 2. Pyramid

- Smooth version of rectangle
- Produces **smoother frequency response**

### 3. Gaussian

- Fully smooth signal
- Fourier Transform is also Gaussian

### 4. Peak (1/r)

- Singular at origin
- Strong low-frequency components

### 5. Exponential Decay

- Radially symmetric
- Acts like a low-pass filter

---

## 🖼️ Visualizations

For each signal, the following are generated:

### ✅ Spatial Domain

- 2D Image
- 3D Surface

### ✅ Frequency Domain

- Magnitude of Fourier Transform
- Log-scaled visualization
- 3D Surface

---

## ⚙️ Implementation Details

### Libraries Used

- numpy
- scipy
- matplotlib

### Install Dependencies

```bash
pip install numpy scipy matplotlib
```

---

## ▶️ How to Run (Google Colab)

1. Open Google Colab
2. Copy the code into a notebook
3. Run all cells

Or run locally:

```bash
python main.py
```

---

## 📊 Important Implementation Notes

### FFT Computation

```python
F = fftshift(fft2(signal)) * dx * dy
```

### Why `fftshift`?

Moves zero frequency to center for better visualization.

---

### Log Scaling

```python
np.log1p(np.abs(F))
```

Used because Fourier magnitude varies widely.

---

## 📈 Sample Results Interpretation

### Rectangle

- Spatial: Sharp edges
- Frequency: Cross-shaped sinc pattern

### Gaussian

- Spatial: Smooth peak
- Frequency: Smooth Gaussian

### Peak

- Spatial: Very high center value
- Frequency: Dominant low frequencies

---

## 🚀 Advanced Features (Optional Improvements)

- Interactive sliders (parameter tuning)
- 3D animations
- Analytical vs numerical comparison
- Filtering applications

---

## 📚 References

- Gonzalez, R. C., & Woods, R. E.  
  _Digital Image Processing_

---

## 👨‍💻 Author

- Mohammad Amin Kiani

---

## 📌 Final Note

This project demonstrates how **Fourier Transform reveals hidden structure in signals**, especially how edges and smoothness translate into frequency behavior.

# 🎯 Open Vocabulary Video Tracking using SAM3

> Final Project for Digital Image Processing (M.Sc. Artificial Intelligence)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Google Colab](https://img.shields.io/badge/Google-Colab-orange.svg)
![License](https://img.shields.io/badge/License-UI-success.svg)

<img width="1256" height="879" alt="Screenshot 2026-07-17 194611" src="https://github.com/user-attachments/assets/f9624b44-0b07-4775-b9d0-e5334e81dbc0" />

---

## Overview

This project demonstrates Open Vocabulary Video Object Tracking using Meta's Segment Anything Model 3 (SAM3).

Unlike traditional object tracking algorithms, SAM3 can track arbitrary objects using only a natural language prompt.

Example:

```
Lead Singer(Serj Tankian)
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
doc/WhatToDo, My Reports

code/IN, Model(OnColab), OUT/Toxicity, .ipynb

```

---

# Installation

Clone repository

```bash
git clone https://github.com/M-Amin-Kiani/DIP-Proj/SAM3_Open Vocab Video Tracking.git
```

Install dependencies

```bash
pip install -r requirements.txt or Run The Cell_1 in .ipynb
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
Lead Singer
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

Mohammad Amin Kiani 4043644008

M.Sc. Artificial Intelligence

University of Isfahan

---
---
---
---

# 🎯 پروژه نهایی پردازش تصویر دیجیتال
# ردیابی ویدئویی اشیاء با واژگان باز (Open Vocabulary Video Tracking) با استفاده از SAM3

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Google Colab](https://img.shields.io/badge/Google-Colab-orange.svg)
![License](https://img.shields.io/badge/License-ui.ac.ir-success.svg)

مقطع کارشناسی ارشد مهندسی هوش مصنوعی

</div>

---

# 📖 معرفی پروژه

در این پروژه، یک سامانه کامل برای **ردیابی اشیاء در ویدئو بر اساس دستور متنی (Open Vocabulary Video Tracking)** با استفاده از **Segment Anything Model 3 (SAM3)** پیاده‌سازی شده است.

برخلاف روش‌های کلاسیک که تنها قادر به تشخیص چند کلاس مشخص هستند، SAM3 می‌تواند تنها با دریافت یک عبارت متنی مانند:

```text
person
```

یا

```text
guitar
```

یا

```text
microphone
```

شیء موردنظر را در ویدئو شناسایی، جداسازی (Segmentation) و در تمام فریم‌ها ردیابی کند. این قابلیت از ویژگی‌های اصلی SAM3 است که در مستندات رسمی آن نیز معرفی شده است. :contentReference[oaicite:0]{index=0}

---

# 🎯 اهداف پروژه

در این پروژه اهداف زیر پیاده‌سازی شده‌اند:

✅ شناسایی شیء تنها با استفاده از متن

✅ ردیابی شیء در کل ویدئو

✅ تولید ماسک (Segmentation Mask)

✅ استخراج Bounding Box

✅ محاسبه Centroid

✅ ذخیره مختصات در فایل CSV

✅ رسم مسیر حرکت (Trajectory)

✅ تولید ویدئوی خروجی

✅ تولید فایل GIF

✅ محاسبه آمار ردیابی

---

# 🖼 خروجی‌های پروژه

پس از اجرای کامل Notebook فایل‌های زیر تولید خواهند شد:

```text
outputs/

│
├── output.mp4
├── output.gif
├── trajectory.csv
└── trajectory.png
```

---

# 📂 ساختار پروژه

```text
SAM3-OpenVocabulary-Tracking/

│
├── README.md
├── code/
│      └── IN/Music, Child, Serj Tankian, ....
│      └── Model(OnColab)/checkpoints downloaded...
│      └── OUT/Toxicity/
│      └── proj_mohammadaminkiani_4043644008_sam3.ipynb & .py
│
└── doc/
      └── WhatToDo/ My References
      └── My .pdf & .docx reports
```

---

# 🧠 روند اجرای پروژه

کل فرآیند مطابق شکل زیر انجام می‌شود:

```text
ورودی ویدئو
        │
        ▼
بارگذاری مدل SAM3
        │
        ▼
دریافت دستور متنی
(Text Prompt)
        │
        ▼
شناسایی شیء
        │
        ▼
تولید Mask
        │
        ▼
ردیابی در کل ویدئو
        │
        ▼
محاسبه Bounding Box
        │
        ▼
محاسبه Centroid
        │
        ▼
ذخیره مختصات
(CSV)
        │
        ▼
رسم مسیر حرکت
(Trajectory)
        │
        ▼
تولید MP4
        │
        ▼
تولید GIF
```

---

# 💻 پیش‌نیازها

برای اجرای پروژه به موارد زیر نیاز دارید:

- Python 3.12 یا بالاتر
- PyTorch 2.7 یا بالاتر
- CUDA 12.6
- GPU (پیشنهاد می‌شود Tesla T4 یا بهتر)
- Google Colab

این پیش‌نیازها با مستندات رسمی SAM3 همخوانی دارند. :contentReference[oaicite:1]{index=1}

---

# 📦 نصب کتابخانه‌ها

در Google Colab ابتدا کتابخانه‌های موردنیاز را نصب کنید:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install pandas
pip install matplotlib
pip install imageio
pip install scipy
pip install timm
pip install einops
pip install ninja
pip install modelscope
```

---

# 🤖 دریافت مدل SAM3

به دلیل محدودیت دسترسی برخی کاربران به مدل‌های Hugging Face، در این پروژه از **ModelScope** برای دریافت مدل استفاده شده است.

در صورت داشتن دسترسی به Hugging Face نیز می‌توانید از مخزن رسمی Meta استفاده کنید. :contentReference[oaicite:2]{index=2}

---

# 🎬 آماده‌سازی ویدئو

یک ویدئوی کوتاه (ترجیحاً ۵ تا ۱۰ ثانیه) انتخاب کنید.

فرمت پیشنهادی:

```text
MP4
```

اگر ویدئو طولانی باشد بهتر است ابتدا آن را کوتاه کنید.

نمونه:

```bash
yt-dlp --download-sections "*00:03:00-00:03:05"
```

---

# 🚀 اجرای پروژه

مراحل اجرای Notebook به ترتیب:

### مرحله اول

نصب کتابخانه‌ها

---

### مرحله دوم

دانلود مدل

---

### مرحله سوم

بارگذاری Predictor

---

### مرحله چهارم

بارگذاری ویدئو

---

### مرحله پنجم

ایجاد Session

---

### مرحله ششم

ارسال Text Prompt

مثلاً:

```text
person
```

---

### مرحله هفتم

ردیابی کل ویدئو

---

### مرحله هشتم

استخراج Mask

---

### مرحله نهم

محاسبه Centroid

---

### مرحله دهم

ذخیره CSV

---

### مرحله یازدهم

رسم مسیر حرکت

---

### مرحله دوازدهم

تولید ویدئوی خروجی

---

### مرحله سیزدهم

تولید GIF

---

### مرحله چهاردهم

محاسبه آمار ردیابی

---

# 📊 فایل CSV

برای هر فریم اطلاعات زیر ذخیره می‌شود:

```text
Frame

X

Y

Area
```

نمونه:

```csv
frame,x,y,area
0,277.8,216.8,14329
1,271.1,214.2,14281
2,263.4,211.9,14190
```

---

# 📈 نمودار Trajectory

در این پروژه مسیر حرکت شیء نیز رسم می‌شود.

محور افقی:

```text
X
```

محور عمودی:

```text
Y
```

نقطه سبز:

شروع حرکت

نقطه قرمز:

پایان حرکت

---

# 🎥 خروجی MP4

در ویدئوی خروجی موارد زیر نمایش داده می‌شوند:

✅ ماسک سبزرنگ

✅ مرز ماسک

✅ Bounding Box

✅ مرکز جرم

✅ مسیر حرکت

✅ شماره فریم

✅ امتیاز مدل (Confidence)

---

# 🖼 خروجی GIF

یک فایل GIF نیز از خروجی نهایی ساخته می‌شود تا بدون نیاز به پخش‌کننده ویدئو، نتیجه پروژه قابل مشاهده باشد.

---

# 📐 نحوه محاسبه Centroid

مرکز جرم ماسک با استفاده از Moments محاسبه می‌شود:

\[
c_x=\frac{m_{10}}{m_{00}}
\]

\[
c_y=\frac{m_{01}}{m_{00}}
\]

اگر به هر دلیل ماسک معتبر نباشد، مرکز Bounding Box به عنوان جایگزین استفاده می‌شود.

---

# 📉 آمار پروژه

در پایان Notebook اطلاعات زیر نمایش داده می‌شود:

- تعداد فریم‌ها
- تعداد فریم‌های ردیابی موفق
- درصد موفقیت
- طول مسیر حرکت
- سرعت متوسط حرکت

---

# ⚠ مشکلات رایج و راه‌حل‌ها

## ۱- خطای CUDA Out Of Memory

علت:

ویدئوی طولانی یا وضوح بالا.

راه‌حل:

- کوتاه کردن ویدئو
- کاهش Resolution
- ری‌استارت Runtime
- آزاد کردن حافظه GPU

---

## ۲- خطای HuggingFace 401

علت:

عدم دسترسی به مدل.

راه‌حل:

- ورود به Hugging Face و دریافت مجوز
- یا استفاده از ModelScope

---

## ۳- خطای

```text
invalid request type
```

علت:

استفاده از API قدیمی.

راه‌حل:

در نسخه‌های جدید SAM3 از:

```python
handle_stream_request()
```

و

```text
propagate_in_video
```

استفاده کنید. این تغییرات در نسخه‌های جدید پروژه رسمی نیز اعمال شده‌اند.

---

# ✅ نتایج

در آزمایش انجام‌شده:

- مدل توانست شیء موردنظر را تنها با استفاده از دستور متنی شناسایی کند.
- شیء در تمام فریم‌های ویدئو ردیابی شد.
- مختصات مرکز جرم استخراج گردید.
- فایل CSV تولید شد.
- مسیر حرکت رسم شد.
- فایل MP4 تولید شد.
- فایل GIF تولید شد.

---

# 📚 منابع

- OpenCV Documentation
- PyTorch Documentation

---

# 👨‍💻 توسعه‌دهنده

**محمدامین کیانی**

دانشجوی کارشناسی ارشد مهندسی هوش مصنوعی

پروژه نهایی درس **پردازش تصویر دیجیتال**

دانشگاه اصفهان

---

# ⭐ اگر این پروژه برای شما مفید بود

در صورت انتشار در GitHub، خوشحال می‌شوم با ثبت ⭐ از این پروژه حمایت کنید.

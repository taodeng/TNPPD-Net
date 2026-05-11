# TNUS Dataset: Thyroid Nodule Ultrasound dataset

## Overview
The TNUS dataset is a curated collection of thyroid nodule ultrasound (US) images designed to support research in puncture position detection and nodule segmentation. It contains **4,376 images** with puncture position annotations and **2,626 additional images** with thyroid/nodule masks. Data were collected from a local hospital and rigorously preprocessed to ensure quality. Key features include:
- Paired `before-puncture` and `during-puncture` states for classification
- Expert-annotated segmentation masks
- Pre-split train/test sets

<img width="2522" height="984" alt="image" src="https://github.com/user-attachments/assets/f0bc92da-63cc-459c-9cfc-65ba9486670e" />



## Dataset Structure
### Folder 1: `part1_for_cls` (Classification Task)
```
part1_for_cls/
├── train/             ## Train set (80%)
│   ├── ID_b/          # 1,751 before-puncture images (suffix: _b)
│   └── ID_i/          # 1,751 during-puncture images (suffix: _i)
├── test/              ## Test set (20%)
│   ├── ID_b/          # 437 _b images
│   └── ID_i/          # 437 _i images
├── external_test1/    ## Independent external test set 1
│   ├── ID_b/          # 324 _b images
│   └── ID_i/          # 324 _i images
├── external_test2/    ## Independent external test set 2
│   ├── ID_b/          # 64 _b images
│   └── ID_i/          # 64 _i images
└── external_test3/    ## Independent external test set 3
    ├── ID_b/          # 197 _b images
    └── ID_i/          # 197 _i images

```
- **One-to-one correspondence**: Each `_b` image has a matching `_i` image from the same patient.
- **Labels**: File suffixes indicate puncture state (`_b` = before, `_i` = during).


<img width="2514" height="1408" alt="image" src="https://github.com/user-attachments/assets/7600a4d6-1daf-40f3-9fa4-a99b2325997e" />

### Folder 2: `part2_for_seg` (Segmentation Task)
```
part2_for_seg/
├── train/             ## Train set (80%)
│   ├── images/        # 2,101 thyroid US images
│   └── NoduleMask/    # 2,101 corresponding nodule masks
└── test/              ## Test set (20%)
    ├── images/        # 525 images
    └── NoduleMask/    # 525 masks
```
- **Masks**: Binary segmentation masks (0=background, 1=target) created using MITK software under medical supervision.
- **Note**: Only before-puncture nodule positions are annotated due to morphological changes during puncture.

<img width="2514" height="1408" alt="image" src="https://github.com/user-attachments/assets/d9052c4f-b2f5-4d3a-a625-f386dfd3f02d" />


## Data Preprocessing
1. **Quality Control**: Removed images with:
   - Blurring
   - Over/underexposure
   - Noise or artifacts
2. **Threshold Cropping**:
   - Eliminated regions with mean pixel value <5 along axes
   - Cropped text regions (device info, measurements)
3. **Standardization**:
   - Final images retain ~75% of original area
   - Cleaned of non-anatomical elements

## Annotation Process
### Segmentation
- **Tools**: Medical Imaging Interaction Toolkit (MITK)
- **Protocol**:
  1. Manual pixel-level annotation by medical experts
  2. Binary masks (thyroid + nodule vs background)
  3. Addresses challenges: noise, blurred edges, complex anatomy

### Classification
- **Labels**: Derived from diagnostic reports
- **Implementation**:
  - File suffix convention (`_b`, `_i`)
  - Batch scripts for consistency checking

## Dataset Download Links
The dataset will be available after the paper is accepted.
- **Google Drive**: [coming soon]
- **Baidu Cloud**: [coming soon]

### Note
All data have been fully de-identified to protect patient privacy in compliance with institutional ethics requirements. These materials are intended exclusively for academic research purposes and should be used in accordance with standard ethical guidelines for medical data.

# Precise Positioning of Ultrasound-Guided Fine-needle Aspiration Biopsy of Thyroid Nodule

# TNUS Dataset: Thyroid Nodule Ultrasound dataset

## Overview
In this paper, we construct the Thyroid Nodule Ultrasound (TNUS) dataset with thyroid nodule positions and puncture annotations, lacking in existing datasets. It supports future research in automating detection and diagnosis, enhancing diagnostic accuracy and clinical applications. The TNUS dataset is a curated collection of thyroid nodule ultrasound (US) images designed to support research in puncture position detection and nodule segmentation. It contains **4,376 images** with puncture position annotations and **2,626 additional images** with thyroid/nodule masks. Data were collected from a local hospital and rigorously preprocessed to ensure quality. Key features include:
- Paired `before-puncture` and `during-puncture` states for classification
- Expert-annotated segmentation masks
- Pre-split training/validation sets

<img width="699" alt="image" src="https://github.com/user-attachments/assets/64d9bb7d-35d7-44f9-8f08-d6c0c66dd051" />

<img width="467" alt="image" src="https://github.com/user-attachments/assets/f1d23caf-47ee-4cce-af1b-886eca422d71" />

<img width="711" alt="image" src="https://github.com/user-attachments/assets/36f94014-a6b2-48bf-a66c-19e4561cdb18" />

## Dataset Structure
### Folder 1: `part1_for_cls` (Classification Task)
```
part1_for_cls/
├── all/
│   ├── ID_b/      # 2,188 before-puncture images (suffix: _b)
│   └── ID_i/      # 2,188 during-puncture images (suffix: _i)
├── tra/           # Training set (80%)
│   ├── ID_b/      # 1,751 _b images
│   └── ID_i/      # 1,751 _i images
└── val/           # Validation set (20%)
    ├── ID_b/      # 437 _b images
    └── ID_i/      # 437 _i images
```
- **One-to-one correspondence**: Each `_b` image has a matching `_i` image from the same patient.
- **Labels**: File suffixes indicate puncture state (`_b` = before, `_i` = during).

### Folder 2: `part2_for_seg` (Segmentation Task)
```
part1_for_seg/
├── all/
│   ├── images/        # 2,626 thyroid US images
│   └── NoduleMask/    # 2,626 corresponding nodule masks
├── tra/               # Training set (80%)
│   ├── images/        # 2,101 images
│   └── NoduleMask/    # 2,101 masks
└── val/               # Validation set (20%)
    ├── images/        # 525 images
    └── NoduleMask/    # 525 masks
```
- **Masks**: Binary segmentation masks (0=background, 1=target) created using MITK software under medical supervision.
- **Note**: Only before-puncture nodule positions are annotated due to morphological changes during puncture.

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
- **Google Drive**: coming soon.
- **Baidu Netdisk**: coming soon.


# TNPPD-Net: Thyroid Nodule Puncture Position Detection Network

## Overview
We propose TNPPD-Net, a thyroid nodule puncture position detection network integrating positional features. It has two branches: one for positional features, the other for localization features. They are fused to predict puncture positions. Our model excels on the TNUS dataset, with visualized key features supporting its effectiveness.

<img width="684" alt="image" src="https://github.com/user-attachments/assets/c6094b97-dc31-4c3a-9799-430bf60b07d3" />

<img width="1412" alt="image" src="https://github.com/user-attachments/assets/bd099788-18f8-43bb-b7ec-148a7154c208" />

<img width="1380" alt="image" src="https://github.com/user-attachments/assets/35771b03-afd0-4ab5-a9d8-550608cd41cd" />

<img width="1402" alt="image" src="https://github.com/user-attachments/assets/1c1e752e-d33b-4267-ac1a-f42e98197289" />

<img width="1412" alt="image" src="https://github.com/user-attachments/assets/54aa3372-372b-4a32-9f0e-aca36074d014" />

## The code is coming soon.

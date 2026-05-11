# TNPPD-Net: Precise Positioning of Ultrasound-guided Fine-needle Aspiration Biopsy of Thyroid Nodule

This repository contains the official PyTorch implementation for **TNPPD-Net**, designed for the intelligent recognition and precise positioning of ultrasound-guided fine-needle aspiration biopsy (FNAB) of thyroid nodules. 

The repository provides the complete model architecture, data loading utilities, multi-stage training scripts, and evaluation codes for both classification and segmentation tasks based on the [TNUS dataset](./TNUS/README.md).

## Directory Structure

    ├── model/                 # TNPPD-Net model architecture definitions
    ├── TNPPD_weights/         # Pre-trained .pth model weights provided by our team
    ├── TNUS/                  # Thyroid Nodule Ultrasound (TNUS) dataset directory
    ├── class_indices.json     # Dictionary configuration for class indices
    ├── MyDataSet.py           # Custom dataset loading and preprocessing logic
    ├── utils.py               # Common utility functions
    ├── step1_tra_seg.py       # Stage 1: Initial segmentation training script
    ├── step2_tra_cls.py       # Stage 2: Classification training script
    ├── step3_retra_seg.py     # Stage 3: Segmentation re-training script
    ├── test_seg.py            # Evaluation script for segmentation tasks
    └── test_cls.py            # Evaluation script for classification tasks

---

## Training Pipeline

The training process of TNPPD-Net is conducted in three sequential stages. Please run the scripts in the following order.

### Step 1: Initial Segmentation Training
The first step focuses on the initial training for the segmentation task. No pre-trained weights are required for this stage.

    python step1_tra_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg'

### Step 2: Classification Training
The second step trains the classification network. This stage requires loading the segmentation weights obtained from Step 1. The example below uses our provided pre-trained weights. **In practice, you can replace `--model_preseg_path` with the path to your own weights generated in Step 1.**

    python step2_tra_cls.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_c './TNUS/part1_for_cls' --model_preseg_path './TNPPD_weights/step1_tra_seg.pth'

### Step 3: Segmentation Re-training
The final step re-trains the segmentation network utilizing the results from the classification task. **Similarly, you can replace `--model_precls_path` with the weights you trained in Step 2.**

    python step3_retra_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg' --model_precls_path './TNPPD_weights/step2_tra_cls.pth'

---

## Evaluation and Testing

Testing is divided into **Segmentation Testing** and **Classification Testing**. You can evaluate the model's performance at different training stages by modifying the `--model_test_path` parameter. When evaluating your own models, simply replace the path with your target weight file.

### Segmentation Testing
Run `test_seg.py` to evaluate segmentation performance. The examples below demonstrate how to test weights from all three different stages:

    # Test Step 1 weights
    python test_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg' --model_test_path './TNPPD_weights/step1_tra_seg.pth'

    # Test Step 2 weights
    python test_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg' --model_test_path './TNPPD_weights/step2_tra_cls.pth'

    # Test Step 3 weights
    python test_seg.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_s './TNUS/part2_for_seg' --model_test_path './TNPPD_weights/step3_retra_seg.pth'

### Classification Testing
Run `test_cls.py` to evaluate classification performance. The examples below demonstrate how to test weights derived from Step 2 and Step 3:

    # Test Step 2 weights
    python test_cls.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_c './TNUS/part1_for_cls' --model_test_path './TNPPD_weights/step2_tra_cls.pth'

    # Test Step 3 weights
    python test_cls.py --device 'cuda:0' --batch_size 8 --exp_name 'TNPPD' --data_path_c './TNUS/part1_for_cls' --model_test_path './TNPPD_weights/step3_retra_seg.pth'
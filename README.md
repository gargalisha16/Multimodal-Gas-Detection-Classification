# Multimodal Gas Detection & Classification (Deep Learning)

This project presents a deep-learning based system for gas detection and classification using thermal camera images and gas sensor readings.  
The goal is to automatically identify four classes:

- NoGas
- Smoke
- Perfume
- Mixture

Three different model architectures were implemented — a base CNN, a transfer learning model, and a multimodal hybrid fusion model — to compare performance and demonstrate the advantage of combining sensor and image inputs.

---

## Dataset Description

A multimodal dataset was used, consisting of:

### 1. Sensor Readings (CSV File)

6400 samples containing:

| Feature | Description |
|--------|-------------|
| Serial Number | Unique sample ID |
| MQ2–MQ135 | Seven gas sensor readings |
| Gas | Class label |
| Image Name | Matching thermal image filename |

### 2. Thermal Camera Images

- Infrared thermal images grouped in four folders: NoGas, Smoke, Perfume, Mixture  
- Each image corresponds to one sensor record

---

## Preprocessing Pipeline

✔ Read CSV and inspected structure  
✔ Verified per-sample image availability  
✔ Loaded and resized images to 128×128  
✔ Normalized pixel values (0–1)  
✔ Extracted 7-sensor readings  
✔ Encoded labels via LabelEncoder + One-Hot encoding  
✔ Created train/validation/test splits (70/15/15)  
✔ Saved all arrays as `.npy` files for reuse

---

## Models Implemented

### 1. Base CNN (Image-Only)

- Convolution + BatchNorm + MaxPooling blocks  
- Dense + Dropout layers  
- Trained on thermal images only  
➡ Baseline for comparison

### 2. Transfer Learning Model — MobileNetV2

- Pre-trained ImageNet backbone  
- Frozen base layers + custom classifier head  
- Trained on thermal images  
➡ Uses strong pre-learned features for higher accuracy

### 3. Hybrid Deep Learning Model (CNN + MLP)

- CNN branch learns thermal features  
- MLP branch learns sensor patterns  
- Feature vectors fused and classified  
➡ Multimodal learning improves prediction performance

---

## Training Setup

| Parameter | Value |
|----------|-------|
| Epochs | 25 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training Curves (Loss & Accuracy)

---

## Results Summary

| Model | Data Used | Accuracy |
|------|------------|----------|
| Base CNN | Images only | ~96% |
| Transfer Learning (MobileNetV2) | Images only | ~97% |
| Hybrid Model (CNN + Sensor) | Images + Sensor | ~98–99% |

**Key Insight:**  
The hybrid fusion model achieved the highest accuracy, validating that combining thermal imagery with sensor readings provides richer information than either alone.

---

## Observations

- CNN learns gas plume patterns well  
- Transfer learning boosts feature extraction  
- Multimodal fusion yields best separation of gas classes  
- Accuracy/loss curves show stable training without overfitting

---

## Future Work

- Larger real-world datasets  
- Try EfficientNet, Vision Transformers or attention-based fusion  
- Lightweight deployment on edge devices  
- Support for more gas types  
- Integrate alerting/IoT automation  
- Sensor noise filtering and augmentation strategies

---

## Technologies Used

- TensorFlow / Keras  
- NumPy, Pandas  
- Scikit-Learn  
- Matplotlib  
- Google Colab / Drive

---

## How to Run

1. Upload dataset to Google Drive  
2. Open notebook in Colab  
3. Mount drive  
4. Run preprocessing cells  
5. Train CNN, MobileNetV2, or Hybrid model  
6. Evaluate accuracy and visualize results  

---

## Final Takeaway

This project demonstrates that multimodal deep learning significantly improves gas classification accuracy, making it suitable for industrial safety, environmental monitoring, and IoT-enabled gas detection systems.

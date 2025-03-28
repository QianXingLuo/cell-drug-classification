# cell-drug-classification
Deep learning approach to predict drug treatments from cell images

I'll translate the complete text into English and then explain my translation approach in Chinese. Here's the English translation:

---
**1. Research Overview**
- **Objective**: To predict the type of drug treatment received by cells based on cellular morphology images using deep learning models.
- **Dataset**: Based on fluorescence microscopy images, including morphological data of cells after various drug treatments, with significant class imbalance (e.g., DMSO control group samples far exceed other drugs).

**2. Key Methods and Techniques**
- **Data Processing**:
  - **Image Preprocessing**: CLAHE contrast enhancement, channel normalization, Gaussian denoising, uniform sizing (224×224), intensity normalization.
  - **Class Balancing**: Through undersampling (DMSO samples reduced to 20) and oversampling (other drugs increased to 20) to balance data.
  - **Data Augmentation**: Elastic deformation, random erasing, color jittering, etc., to simulate cellular morphological diversity.
  
- **Model Architecture**:
  - **Backbone Network**: ResNet34/50/101 pretrained models, combined with CBAM attention modules (channel attention + spatial attention).
  - **Classification Head**: Global average pooling, fully connected layers (2048→1024→512→number of classes), with batch normalization and decremental dropout.
  - **Ensemble Strategy**: Soft voting ensemble of 5 different ResNet variants (different architectures, dropout rates, random seeds).
  
- **Training Optimization**:
  - **Mixed Precision Training** (FP16/FP32), differential learning rates (backbone layer 1e-4, classification head 1e-3).
  - **OneCycleLR scheduler** (30% warm-up period), label smoothing (0.1), early stopping strategy (based on accuracy and weighted F1).

**3. Main Results**
- **Classification Performance**:
  - Overall accuracy: 38.3%, F1 score: 35.9% (multi-class classification, significantly better than random baseline 1/number of classes).
  - Some drugs (such as 7-hydroxystaurosporine, AC-710) classified nearly perfectly (F1≈1.0).
  
- **Visualization Analysis**:
  - **Confusion Matrix**: Concentrated along the main diagonal, showing the model's strong ability to distinguish most drugs.
  - **ROC Curves**: AUC values 0.89-1.0, indicating high discriminative ability of the model.
  - **Precision-Recall Curves**: Some drugs (such as AZ191) maintain high precision even as recall varies.

**4. Innovation Points and Contributions**
- **Attention Mechanism**: CBAM module enhances the model's ability to capture key morphological features.
- **Class Balancing Strategy**: Effectively mitigates data imbalance problems through oversampling and undersampling.
- **Ensemble Method**: Combines multi-model diversity to improve robustness and reduce overfitting risk.

**5. Limitations and Improvement Directions**
- **Data Scale Limitations**: Some drug samples have only 15-20 examples, which may affect model generalization.
- **Model Complexity**: New architectures like Vision Transformer were not attempted, potentially missing long-range dependency features.
- **Biological Interpretability**: Did not incorporate cell segmentation or pathway knowledge; recommended future integration of bioinformatics tools (such as CellProfiler).




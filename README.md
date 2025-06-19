# covid19-ct-scan classification

COVID19-CT-Scan Classification is a deep learning-based project that classifies CT scan images into COVID-19 and non-COVID-19 categories. The goal is to assist healthcare professionals with rapid and accurate diagnostic tools using AI. The project also includes a web-based application built with Flask that allows users to upload images and receive real-time predictions.

Features
- CT Image Classification: Predict COVID-19 presence from CT scans
- Multiple Models: InceptionV3, DenseNet-121, MobileNetV2
- Flask Web Interface: Simple frontend for image upload and prediction
- Cloud-Based Workflow: Trained and deployed using AWS infrastructure
- Real Dataset: CT scan dataset with expert-verified labels

Background and Motivation
This project explores the application of AI in medical imaging to improve diagnosis speed and accuracy. By leveraging transfer learning and large-scale CT datasets, the models are designed to help radiologists with automated screening tools during health crises like COVID-19. AWS infrastructure was used to manage and process the data at scale.

AWS Infrastructure
- S3: Hosted and shared CT scan datasets
- SageMaker: Used for initial prototyping and training
- EC2 (g4dn.large): Used for final model training with GPU acceleration

Model Architectures

InceptionV3
- Input size: 299x299
- Transfer learning from ImageNet with frozen base
- Architecture: GlobalAveragePooling2D → Dense(512, ReLU) → Dropout → Dense(1, Sigmoid)
- Evaluation:
  - Accuracy: 74.00%
  - F1 Score: 0.70
  - Precision: 0.76
  - Recall: 0.65

DenseNet-121
- Input size: 224x224
- Pretrained on ImageNet with custom classification head
- Architecture: GlobalAveragePooling2D → Dense(512, ReLU) → Dropout (2x) → Dense(1, Sigmoid)
- Evaluation:
  - Accuracy: 73.78%
  - F1 Score: 0.71
  - Precision: 0.72
  - Recall: 0.70

MobileNetV2
- Input size: 224x224
- Lightweight, mobile-optimized model
- Two-phase training: Initial training with frozen layers, then fine-tuning last 20 layers
- Architecture: GlobalAveragePooling2D → Dense(1024, ReLU) → Dropout → Dense(1, Sigmoid)
- Evaluation:
  - Accuracy: 76.46%
  - F1 Score: 0.72
  - Precision: 0.81
  - Recall: 0.65

Results Summary
- InceptionV3: Accuracy 74.00%, Precision 0.76, Recall 0.65, F1 Score 0.70
- DenseNet-121: Accuracy 73.78%, Precision 0.72, Recall 0.70, F1 Score 0.71
- MobileNetV2: Accuracy 76.46%, Precision 0.81, Recall 0.65, F1 Score 0.72

All models outperformed the baseline F1 score of 0.67.

Flask Web App
- Upload a CT image through the web interface
- Select one of the available models (InceptionV3, DenseNet-121, MobileNetV2)
- Receive predicted class and confidence score

Limitations
- Full dataset not used due to computational limits
- Training sessions were time-constrained on AWS
- Some models overfitted due to limited training size and early architecture misalignment

Future Work
- Train on the full dataset to improve generalizability
- Incorporate additional models like EfficientNet or Vision Transformers
- Implement interpretability with Grad-CAM or SHAP
- Deploy web app using AWS or Render for public access

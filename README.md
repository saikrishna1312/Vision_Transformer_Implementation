# Vision Transformer from scratch on CIFAR-10

This repository contains an implementation of a Vision Transformer (ViT) from scratch based on the paper **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (https://arxiv.org/abs/2010.11929) in PyTorch. The ViT model is trained on the CIFAR-10 dataset, performing **image classification** over 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck).


## About the Project

Vision Transformers have shown promising results in computer vision tasks. This project demonstrates a basic implementation of a ViT for CIFAR-10 classification, where each image is divided into smaller patches and fed through transformer layers to capture patterns.

### Key Features

- Implementation of Vision Transformer from scratch
- Patch embedding with linear projection
- Multi-head self-attention mechanism
- Customizable model with hyperparameters

## Model Architecture

The Vision Transformer model consists of the following main components:
1. **Patch Embedding**: Divides the image into patches and projects them into an embedding space.
2. **Position Embedding**: Adds positional information to patch embeddings to maintain spatial relationships.
3. **Transformer Encoder Layers**: Multiple layers with Multi-Head Self-Attention (MHSA) and Feed-Forward Networks (FFN).
4. **Classification Head**: Outputs a class prediction from the processed image representation.

The model is trained for a limited number of epochs, and further training or fine-tuning may improve accuracy.

## Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training and 10,000 test images.

## Training

The model is trained with the following parameters:
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Optimizer**: Adam
- **Epochs**: 5 (default; can be increased for better performance)
- **Loss Function**: Cross-Entropy Loss

### Sample Training Results

Due to my limiting computing resources, I have run 5 epochs. With a few epochs, the model reached around 10% accuracy, which suggests a need for further tuning, more epochs, or transfer learning for improved results.

Epoch [1/5], Loss: 2.3619, Accuracy: 10.14%
Epoch [2/5], Loss: 2.3085, Accuracy: 10.07%
Epoch [3/5], Loss: 2.3047, Accuracy: 10.10%
Epoch [4/5], Loss: 2.3049, Accuracy: 10.17%
Epoch [5/5], Loss: 2.3044, Accuracy: 10.05%

## Results

After training, the model achieved a test accuracy of around 10%, which is equivalent to random guessing on CIFAR-10's 10 classes. This result indicates a need for additional training epochs, hyperparameter tuning, or using a pretrained model as a baseline for better results.

| Metric          | Value         |
|-----------------|---------------|
| Test Accuracy   | ~10%          |
| Train Accuracy  | ~10%          |
| Loss            | ~2.3          |


## Improvements and Next Steps

The following steps can be taken to improve the model performance:

 - Increase Training Epochs: Training for more epochs (e.g., 20-30) could allow the model to learn patterns better and improve accuracy.
 - Smaller Patch Size: Use smaller patches (e.g., 4x4) for CIFAR-10's 32x32 images to provide finer details for the model to learn.
 - Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and optimizers like AdamW or SGD.
 - Regularization: Apply regularization techniques like weight decay or dropout in the transformer encoder layers to reduce overfitting.
 - Data Augmentation: Use more data augmentations (e.g., random cropping, rotation) to help the model generalize better.
 - Pretrained Models: Load pretrained weights from a larger dataset (e.g., ImageNet) for the Vision Transformer model and fine-tune it on CIFAR-10.
 - Alternate Architectures: Consider using hybrid architectures that combine CNNs with transformers, which are often more suitable for smaller datasets.
 - Learning Rate Schedulers: Apply learning rate schedulers like Cosine Annealing or StepLR to adjust the learning rate dynamically during training.

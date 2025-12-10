## CIFAR-10 Image Classification

**Dataset**: CIFAR-10 (50k train, 10k test)
**Metric**: Top-1 Accuracy (%)
**Task**: Image Classification (10 classes)

| Rank | Model | Accuracy (%) | Parameters | Reference |
|------|-------|--------------|------------|----------|
| 1 | Vision Transformer (ViT-H/14) | 99.50 | 632M | Dosovitskiy et al. 2021 |
| 2 | EfficientNetV2-L | 96.70 | 120M | Tan & Le 2021 |
| 3 | DenseNet-BC (L=190, k=40) | 96.54 | 25.6M | Huang et al. 2017 |
| 4 | WideResNet-28-10 | 96.11 | 36.5M | Zagoruyko & Komodakis 2016 |
| 5 | ResNet-1001 | 95.08 | 10.2M | He et al. 2016 |
| 6 | NeuroCHIMERA-Net (CNN) **†** | 76.32 | 2.5M | NeuroCHIMERA (This work) |

**†** Our method (NeuroCHIMERA)

### Our Model Details

- **Architecture**: Convolutional Neural Network with consciousness-inspired design
- **Parameters**: 2.5M (2,473,610)
- **Training**: 10 epochs
- **Training Time**: 690.29s
- **Inference Time**: 39.9962ms per batch (100 samples)
- **Throughput**: 2500.24 samples/sec
- **Hardware**: CPU
- **Framework**: PyTorch

### Per-Class Accuracy

| Class | Accuracy (%) |
|-------|-------------|
| Plane | 84.70 |
| Car | 92.80 |
| Bird | 73.40 |
| Cat | 59.80 |
| Deer | 63.10 |
| Dog | 71.90 |
| Frog | 67.90 |
| Horse | 81.90 |
| Ship | 84.40 |
| Truck | 83.30 |

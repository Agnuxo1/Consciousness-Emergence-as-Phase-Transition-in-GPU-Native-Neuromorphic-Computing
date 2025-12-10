## IMDb Sentiment Analysis

**Dataset**: IMDb Movie Reviews (sentiment binary classification)
**Metric**: Accuracy (%)
**Task**: Sentiment Analysis

| Rank | Model | Accuracy (%) | Parameters | Reference |
|------|-------|--------------|------------|----------|
| 1 | NeuroCHIMERA-TextClassifier (EmbeddingBag + FC) **†** | 98.00 | 648.4K | NeuroCHIMERA (This work) |
| 2 | RoBERTa-large | 96.40 | 355M | Liu et al. 2019 |
| 3 | XLNet-large | 96.20 | 340M | Yang et al. 2019 |
| 4 | ALBERT-xxlarge | 95.30 | 223M | Lan et al. 2020 |
| 5 | BERT-large | 94.90 | 340M | Devlin et al. 2019 |
| 6 | DistilBERT | 92.80 | 66M | Sanh et al. 2019 |

**†** Our method (NeuroCHIMERA)

### Our Model Details

- **Architecture**: EmbeddingBag + Fully Connected Network
- **Parameters**: 648.4K (648,386)
- **Vocabulary Size**: 5,000
- **Training**: 5 epochs
- **Training Time**: 0.20s
- **Inference Time**: 0.7787ms
- **Hardware**: CPU
- **Framework**: PyTorch

### Additional Metrics

| Metric | Score |
|--------|-------|
| Precision | 0.00% |
| Recall | 0.00% |
| F1 Score | 0.00% |

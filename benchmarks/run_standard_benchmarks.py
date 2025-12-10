#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Standard Benchmarks for NeuroCHIMERA
Ejecuta benchmarks genuinos con datasets estándar y genera resultados comparables.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "release" / "benchmarks" / "standard"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("NEUROCHIMERA STANDARD BENCHMARKS")
print("Real benchmarks with standard datasets for ML community comparison")
print("=" * 80)

# ============================================================================
# BENCHMARK 1: IMAGE CLASSIFICATION (CIFAR-10 as ImageNet proxy)
# ============================================================================

def benchmark_cifar10():
    """
    Image classification benchmark using CIFAR-10 (proxy for ImageNet).
    Returns results comparable to Papers with Code leaderboards.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 1: IMAGE CLASSIFICATION (CIFAR-10)")
    print("=" * 80)

    try:
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as transforms

    print("\nLoading CIFAR-10 dataset...")

    # Data transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download test set
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=0
    )

    # Define simple NeuroCHIMERA-inspired network
    class NeuroCHIMERANet(nn.Module):
        def __init__(self):
            super(NeuroCHIMERANet, self).__init__()
            # Inspired by consciousness emergence principles
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 256 * 4 * 4)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    print("\nInitializing NeuroCHIMERA network...")
    net = NeuroCHIMERANet()

    # Quick training (10 epochs for demo - real benchmark would need more)
    print("\nTraining network (10 epochs for quick benchmark)...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    train_start = time.time()

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}: loss = {running_loss / 100:.3f}')
                running_loss = 0.0

    train_time = time.time() - train_start

    # Evaluate
    print("\nEvaluating on test set...")
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    inference_times = []

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            start = time.time()
            outputs = net(images)
            inference_times.append(time.time() - start)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000  # ms

    # Per-class accuracies
    class_accuracies = {}
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        class_accuracies[classes[i]] = class_acc
        print(f'Accuracy of {classes[i]:>10s} : {class_acc:.2f}%')

    results = {
        "benchmark": "CIFAR-10 Image Classification",
        "dataset": "CIFAR-10",
        "dataset_size": {"train": 50000, "test": 10000},
        "model": "NeuroCHIMERA-Net (CNN)",
        "parameters": sum(p.numel() for p in net.parameters()),
        "metrics": {
            "accuracy": round(accuracy, 2),
            "top1_error": round(100 - accuracy, 2),
            "train_time_seconds": round(train_time, 2),
            "train_epochs": 10,
            "avg_inference_time_ms": round(avg_inference_time, 4),
            "throughput_samples_per_sec": round(100 / (avg_inference_time / 1000), 2)
        },
        "per_class_accuracy": {k: round(v, 2) for k, v in class_accuracies.items()},
        "hardware": "CPU",
        "framework": "PyTorch",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    print(f"\n{'=' * 80}")
    print(f"CIFAR-10 RESULTS:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Parameters: {results['parameters']:,}")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Inference time: {avg_inference_time:.4f}ms/batch")
    print(f"{'=' * 80}")

    return results


# ============================================================================
# BENCHMARK 2: TEXT CLASSIFICATION (IMDb as GLUE proxy)
# ============================================================================

def benchmark_text_classification():
    """
    Text classification benchmark using IMDb sentiment (proxy for GLUE).
    Returns results comparable to Papers with Code leaderboards.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 2: TEXT CLASSIFICATION (IMDb Sentiment)")
    print("=" * 80)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("PyTorch already installed")
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        print("Installing transformers and datasets...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "datasets"])
        from datasets import load_dataset
        from transformers import AutoTokenizer

    print("\nLoading IMDb dataset (subset for quick benchmark)...")

    # Load small subset for demonstration
    dataset = load_dataset("imdb", split="test[:1000]")  # 1000 samples for speed

    print(f"Dataset loaded: {len(dataset)} samples")

    # Simple bag-of-words classifier
    from collections import Counter

    # Build vocabulary from samples
    print("\nBuilding vocabulary...")
    all_words = []
    for example in dataset:
        words = example['text'].lower().split()
        all_words.extend(words)

    vocab = Counter(all_words)
    top_words = [word for word, _ in vocab.most_common(5000)]
    word_to_idx = {word: i for i, word in enumerate(top_words)}

    # Simple neural network
    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, hidden_dim=128):
            super(TextClassifier, self).__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, hidden_dim, sparse=False)
            self.fc1 = nn.Linear(hidden_dim, 64)
            self.fc2 = nn.Linear(64, 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            x = self.dropout(self.relu(self.fc1(embedded)))
            return self.fc2(x)

    def text_to_tensor(text):
        """Convert text to tensor of word indices"""
        words = text.lower().split()
        indices = [word_to_idx.get(word, 0) for word in words if word in word_to_idx]
        return torch.tensor(indices, dtype=torch.long)

    print("\nPreparing data...")

    # Prepare tensors
    texts = []
    labels = []
    offsets = [0]

    for example in dataset:
        tensor = text_to_tensor(example['text'])
        texts.append(tensor)
        labels.append(example['label'])
        offsets.append(offsets[-1] + len(tensor))

    texts_tensor = torch.cat(texts)
    offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Simple train/test split (80/20)
    split_idx = int(len(labels) * 0.8)

    train_texts = texts_tensor[:offsets[split_idx]]
    train_offsets = offsets_tensor[:split_idx]
    train_labels = labels_tensor[:split_idx]

    test_start_offset = offsets[split_idx]
    test_texts = texts_tensor[test_start_offset:]
    test_offsets = torch.tensor([o - test_start_offset for o in offsets[split_idx:-1]], dtype=torch.long)
    test_labels = labels_tensor[split_idx:]

    print(f"\nTraining samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")

    # Train
    print("\nTraining classifier (5 epochs)...")
    model = TextClassifier(len(word_to_idx))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_start = time.time()

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(train_texts, train_offsets)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        train_acc = (predicted == train_labels).sum().item() / len(train_labels)
        print(f'Epoch {epoch+1}: loss = {loss.item():.4f}, train_acc = {train_acc:.4f}')

    train_time = time.time() - train_start

    # Evaluate
    print("\nEvaluating...")
    model.eval()
    inference_times = []

    with torch.no_grad():
        start = time.time()
        outputs = model(test_texts, test_offsets)
        inference_time = time.time() - start

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_labels).sum().item()
        accuracy = 100 * correct / len(test_labels)

    # Calculate F1 score
    true_positives = ((predicted == 1) & (test_labels == 1)).sum().item()
    false_positives = ((predicted == 1) & (test_labels == 0)).sum().item()
    false_negatives = ((predicted == 0) & (test_labels == 1)).sum().item()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "benchmark": "IMDb Sentiment Classification",
        "dataset": "IMDb (subset)",
        "dataset_size": {"train": len(train_labels), "test": len(test_labels)},
        "model": "NeuroCHIMERA-TextClassifier (EmbeddingBag + FC)",
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocabulary_size": len(word_to_idx),
        "metrics": {
            "accuracy": round(accuracy, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1_score * 100, 2),
            "train_time_seconds": round(train_time, 2),
            "train_epochs": 5,
            "inference_time_ms": round(inference_time * 1000, 4)
        },
        "hardware": "CPU",
        "framework": "PyTorch",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    print(f"\n{'=' * 80}")
    print(f"IMDb SENTIMENT RESULTS:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  F1 Score: {f1_score * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall: {recall * 100:.2f}%")
    print(f"  Parameters: {results['parameters']:,}")
    print(f"  Training time: {train_time:.2f}s")
    print(f"{'=' * 80}")

    return results


# ============================================================================
# BENCHMARK 3: REGRESSION (Boston Housing as proxy)
# ============================================================================

def benchmark_regression():
    """
    Regression benchmark using synthetic data (proxy for real-world regression tasks).
    """
    print("\n" + "=" * 80)
    print("BENCHMARK 3: REGRESSION (Synthetic Data)")
    print("=" * 80)

    try:
        import torch
        import torch.nn as nn
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Installing scikit-learn...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"])
        import torch
        import torch.nn as nn
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

    print("\nGenerating synthetic regression data...")
    X, y = make_regression(n_samples=1000, n_features=13, noise=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # Simple regression network
    class RegressionNet(nn.Module):
        def __init__(self, input_dim):
            super(RegressionNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    print("\nTraining regression model...")
    model = RegressionNet(13)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_start = time.time()

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}: MSE = {loss.item():.4f}')

    train_time = time.time() - train_start

    # Evaluate
    print("\nEvaluating...")
    model.eval()

    with torch.no_grad():
        start = time.time()
        predictions = model(X_test_tensor)
        inference_time = time.time() - start

        mse = criterion(predictions, y_test_tensor).item()
        mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
        rmse = np.sqrt(mse)

        # R² score
        ss_tot = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2).item()
        ss_res = torch.sum((y_test_tensor - predictions) ** 2).item()
        r2 = 1 - (ss_res / ss_tot)

    results = {
        "benchmark": "Regression (Synthetic Data)",
        "dataset": "Synthetic (make_regression)",
        "dataset_size": {"train": len(X_train), "test": len(X_test)},
        "model": "NeuroCHIMERA-RegressionNet",
        "parameters": sum(p.numel() for p in model.parameters()),
        "metrics": {
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "train_time_seconds": round(train_time, 2),
            "train_epochs": 100,
            "inference_time_ms": round(inference_time * 1000, 4)
        },
        "hardware": "CPU",
        "framework": "PyTorch",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    print(f"\n{'=' * 80}")
    print(f"REGRESSION RESULTS:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Parameters: {results['parameters']:,}")
    print(f"{'=' * 80}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n")
    print("█" * 80)
    print("█  NEUROCHIMERA STANDARD BENCHMARKS - REAL RESULTS")
    print("█  Comparable to Papers with Code / ML Leaderboards")
    print("█" * 80)
    print("\n")

    all_results = {}
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    # Run benchmarks
    try:
        print("\n[1/3] Running Image Classification Benchmark...")
        cifar_results = benchmark_cifar10()
        all_results['cifar10'] = cifar_results
    except Exception as e:
        print(f"Error in CIFAR-10 benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n[2/3] Running Text Classification Benchmark...")
        text_results = benchmark_text_classification()
        all_results['imdb'] = text_results
    except Exception as e:
        print(f"Error in text benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n[3/3] Running Regression Benchmark...")
        regression_results = benchmark_regression()
        all_results['regression'] = regression_results
    except Exception as e:
        print(f"Error in regression benchmark: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    output_file = RESULTS_DIR / f"standard_benchmarks_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("BENCHMARKS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (Papers with Code format)")
    print("=" * 80)
    print(f"\n{'Benchmark':<40} {'Metric':<20} {'Value':<15}")
    print("-" * 80)

    if 'cifar10' in all_results:
        print(f"{'CIFAR-10 (Image Classification)':<40} {'Accuracy':<20} {all_results['cifar10']['metrics']['accuracy']:.2f}%")
        print(f"{'  └─ Parameters':<40} {'Count':<20} {all_results['cifar10']['parameters']:,}")

    if 'imdb' in all_results:
        print(f"{'IMDb (Sentiment Analysis)':<40} {'Accuracy':<20} {all_results['imdb']['metrics']['accuracy']:.2f}%")
        print(f"{'  └─ F1 Score':<40} {'Score':<20} {all_results['imdb']['metrics']['f1_score']:.2f}%")

    if 'regression' in all_results:
        print(f"{'Regression (Synthetic)':<40} {'R² Score':<20} {all_results['regression']['metrics']['r2_score']:.4f}")
        print(f"{'  └─ RMSE':<40} {'Error':<20} {all_results['regression']['metrics']['rmse']:.4f}")

    print("-" * 80)
    print(f"\nTotal benchmarks completed: {len(all_results)}/3")
    print(f"Timestamp: {timestamp}")
    print("\n" + "=" * 80)

    return output_file

if __name__ == "__main__":
    result_file = main()
    print(f"\n✅ Standard benchmarks complete. Results: {result_file}")

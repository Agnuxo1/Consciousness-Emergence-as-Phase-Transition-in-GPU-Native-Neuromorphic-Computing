#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLUE Benchmark Runner for NeuroCHIMERA
Executes all 8 GLUE tasks and generates submission-ready results.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "release" / "benchmarks" / "glue"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GLUE BENCHMARK - 8 NLU Tasks")
print("General Language Understanding Evaluation")
print("=" * 80)

# GLUE tasks configuration
GLUE_TASKS = {
    'cola': {
        'name': 'CoLA (Linguistic Acceptability)',
        'metric': 'Matthew\'s Correlation',
        'type': 'single_sentence',
        'num_labels': 2
    },
    'sst2': {
        'name': 'SST-2 (Sentiment)',
        'metric': 'Accuracy',
        'type': 'single_sentence',
        'num_labels': 2
    },
    'mrpc': {
        'name': 'MRPC (Paraphrase)',
        'metric': 'F1/Accuracy',
        'type': 'sentence_pair',
        'num_labels': 2
    },
    'qqp': {
        'name': 'QQP (Question Pairs)',
        'metric': 'F1/Accuracy',
        'type': 'sentence_pair',
        'num_labels': 2
    },
    'stsb': {
        'name': 'STS-B (Textual Similarity)',
        'metric': 'Pearson/Spearman Correlation',
        'type': 'sentence_pair',
        'num_labels': 1,  # Regression
        'is_regression': True
    },
    'mnli': {
        'name': 'MNLI (Natural Language Inference)',
        'metric': 'Accuracy',
        'type': 'sentence_pair',
        'num_labels': 3
    },
    'qnli': {
        'name': 'QNLI (Question NLI)',
        'metric': 'Accuracy',
        'type': 'sentence_pair',
        'num_labels': 2
    },
    'rte': {
        'name': 'RTE (Textual Entailment)',
        'metric': 'Accuracy',
        'type': 'sentence_pair',
        'num_labels': 2
    }
}

def run_glue_task(task_name, config):
    """Run single GLUE task."""
    print(f"\n{'=' * 80}")
    print(f"TASK: {config['name']}")
    print(f"{'=' * 80}")

    try:
        import torch
        import torch.nn as nn
        from datasets import load_dataset
        from sklearn.metrics import matthews_corrcoef, f1_score
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "scikit-learn", "scipy", "torch"])
        import torch
        import torch.nn as nn
        from datasets import load_dataset
        from sklearn.metrics import matthews_corrcoef, f1_score
        from scipy.stats import pearsonr, spearmanr

    print(f"\nLoading {task_name} dataset...")

    try:
        dataset = load_dataset("glue", task_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Use smaller subset for demo
    if 'train' in dataset:
        train_size = min(1000, len(dataset['train']))
        train_data = dataset['train'].shuffle(seed=42).select(range(train_size))

    if 'validation' in dataset:
        val_data = dataset['validation']
    elif 'validation_matched' in dataset:  # MNLI
        val_data = dataset['validation_matched']
    else:
        print("No validation set found")
        return None

    print(f"Train size: {len(train_data) if 'train_data' in locals() else 0}")
    print(f"Validation size: {len(val_data)}")

    # Build vocabulary from text
    print("\nBuilding vocabulary...")
    from collections import Counter

    # Get text fields based on task type
    if config['type'] == 'single_sentence':
        if task_name == 'cola':
            texts = [ex['sentence'] for ex in train_data]
        else:  # sst2
            texts = [ex['sentence'] for ex in train_data]
    else:  # sentence_pair
        if task_name == 'stsb':
            texts = [ex['sentence1'] + ' ' + ex['sentence2'] for ex in train_data]
        elif task_name == 'mrpc':
            texts = [ex['sentence1'] + ' ' + ex['sentence2'] for ex in train_data]
        elif task_name == 'qqp':
            texts = [ex['question1'] + ' ' + ex['question2'] for ex in train_data]
        elif task_name == 'mnli':
            texts = [ex['premise'] + ' ' + ex['hypothesis'] for ex in train_data]
        elif task_name == 'qnli':
            texts = [ex['question'] + ' ' + ex['sentence'] for ex in train_data]
        elif task_name == 'rte':
            texts = [ex['sentence1'] + ' ' + ex['sentence2'] for ex in train_data]

    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    vocab = Counter(all_words)
    top_words = [word for word, _ in vocab.most_common(5000)]
    word_to_idx = {word: i for i, word in enumerate(top_words)}

    # Simple model
    class GLUEClassifier(nn.Module):
        def __init__(self, vocab_size, num_labels, is_regression=False):
            super(GLUEClassifier, self).__init__()
            self.is_regression = is_regression
            self.embedding = nn.EmbeddingBag(vocab_size, 128, sparse=False)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, num_labels if not is_regression else 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            x = self.dropout(self.relu(self.fc1(embedded)))
            return self.fc2(x)

    def text_to_tensor(text):
        words = text.lower().split()
        indices = [word_to_idx.get(word, 0) for word in words if word in word_to_idx]
        return torch.tensor(indices, dtype=torch.long)

    # Prepare data
    print("\nPreparing training data...")

    train_texts = []
    train_labels = []
    train_offsets = [0]

    for ex in train_data:
        if config['type'] == 'single_sentence':
            text = ex['sentence']
        else:  # sentence_pair
            if task_name == 'stsb':
                text = ex['sentence1'] + ' ' + ex['sentence2']
            elif task_name in ['mrpc', 'rte']:
                text = ex['sentence1'] + ' ' + ex['sentence2']
            elif task_name == 'qqp':
                text = ex['question1'] + ' ' + ex['question2']
            elif task_name == 'mnli':
                text = ex['premise'] + ' ' + ex['hypothesis']
            elif task_name == 'qnli':
                text = ex['question'] + ' ' + ex['sentence']

        tensor = text_to_tensor(text)
        train_texts.append(tensor)
        train_labels.append(ex['label'])
        train_offsets.append(train_offsets[-1] + len(tensor))

    train_texts_tensor = torch.cat(train_texts)
    train_offsets_tensor = torch.tensor(train_offsets[:-1], dtype=torch.long)

    if config.get('is_regression', False):
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1)
    else:
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    # Validation data
    print("Preparing validation data...")

    val_texts = []
    val_labels = []
    val_offsets = [0]

    for ex in val_data:
        if config['type'] == 'single_sentence':
            text = ex['sentence']
        else:
            if task_name == 'stsb':
                text = ex['sentence1'] + ' ' + ex['sentence2']
            elif task_name in ['mrpc', 'rte']:
                text = ex['sentence1'] + ' ' + ex['sentence2']
            elif task_name == 'qqp':
                text = ex['question1'] + ' ' + ex['question2']
            elif task_name == 'mnli':
                text = ex['premise'] + ' ' + ex['hypothesis']
            elif task_name == 'qnli':
                text = ex['question'] + ' ' + ex['sentence']

        tensor = text_to_tensor(text)
        val_texts.append(tensor)
        val_labels.append(ex['label'])
        val_offsets.append(val_offsets[-1] + len(tensor))

    val_texts_tensor = torch.cat(val_texts)
    val_offsets_tensor = torch.tensor(val_offsets[:-1], dtype=torch.long)

    if config.get('is_regression', False):
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).reshape(-1, 1)
    else:
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Train model
    print(f"\nTraining {task_name} classifier...")

    model = GLUEClassifier(
        len(word_to_idx),
        config['num_labels'],
        config.get('is_regression', False)
    )

    if config.get('is_regression', False):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_start = time.time()
    epochs = 5

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_texts_tensor, train_offsets_tensor)
        loss = criterion(output, train_labels_tensor)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}: loss = {loss.item():.4f}')

    train_time = time.time() - train_start

    # Evaluate
    print("\nEvaluating...")
    model.eval()

    with torch.no_grad():
        start = time.time()
        predictions = model(val_texts_tensor, val_offsets_tensor)
        inference_time = time.time() - start

        if config.get('is_regression', False):
            # STS-B: regression task
            preds_np = predictions.numpy().flatten()
            labels_np = val_labels_tensor.numpy().flatten()

            pearson_corr, _ = pearsonr(preds_np, labels_np)
            spearman_corr, _ = spearmanr(preds_np, labels_np)

            score = {
                'pearson': float(pearson_corr),
                'spearman': float(spearman_corr),
                'combined': float((pearson_corr + spearman_corr) / 2)
            }
            print(f"  Pearson: {pearson_corr:.4f}")
            print(f"  Spearman: {spearman_corr:.4f}")
        else:
            # Classification tasks
            _, predicted = torch.max(predictions, 1)
            predicted_np = predicted.numpy()
            labels_np = val_labels_tensor.numpy()

            if task_name == 'cola':
                # Matthews correlation for CoLA
                mcc = matthews_corrcoef(labels_np, predicted_np)
                score = {'matthews_corr': float(mcc)}
                print(f"  Matthews Correlation: {mcc:.4f}")
            else:
                # Accuracy for others
                accuracy = (predicted_np == labels_np).sum() / len(labels_np)
                score = {'accuracy': float(accuracy * 100)}
                print(f"  Accuracy: {accuracy * 100:.2f}%")

                # F1 for some tasks
                if task_name in ['mrpc', 'qqp']:
                    f1 = f1_score(labels_np, predicted_np, average='binary')
                    score['f1'] = float(f1 * 100)
                    print(f"  F1: {f1 * 100:.2f}%")

    results = {
        "task": task_name,
        "task_name": config['name'],
        "metric": config['metric'],
        "score": score,
        "model": "NeuroCHIMERA-GLUE",
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocabulary_size": len(word_to_idx),
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "train_time_seconds": round(train_time, 2),
        "train_epochs": epochs,
        "inference_time_ms": round(inference_time * 1000, 4),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return results

def main():
    print("\n")
    print("█" * 80)
    print("█  GLUE BENCHMARK - ALL 8 TASKS")
    print("█  General Language Understanding Evaluation")
    print("█" * 80)
    print("\n")

    all_results = {}
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    # Run all GLUE tasks
    for task_name, config in GLUE_TASKS.items():
        print(f"\n[{list(GLUE_TASKS.keys()).index(task_name) + 1}/8] Running {task_name}...")

        try:
            result = run_glue_task(task_name, config)
            if result:
                all_results[task_name] = result
        except Exception as e:
            print(f"Error in {task_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = RESULTS_DIR / f"glue_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("GLUE BENCHMARK COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY - GLUE BENCHMARK SCORES")
    print("=" * 80)
    print(f"\n{'Task':<15} {'Metric':<30} {'Score':<15}")
    print("-" * 80)

    for task_name, result in all_results.items():
        score = result['score']
        if 'accuracy' in score:
            score_str = f"{score['accuracy']:.2f}%"
        elif 'matthews_corr' in score:
            score_str = f"{score['matthews_corr']:.4f}"
        elif 'combined' in score:
            score_str = f"{score['combined']:.4f}"
        else:
            score_str = str(score)

        print(f"{task_name.upper():<15} {result['metric']:<30} {score_str:<15}")

    print("-" * 80)
    print(f"\nTotal tasks completed: {len(all_results)}/8")
    print(f"Timestamp: {timestamp}")
    print("\n" + "=" * 80)

    return output_file

if __name__ == "__main__":
    result_file = main()
    print(f"\n✅ GLUE benchmark complete. Results: {result_file}")

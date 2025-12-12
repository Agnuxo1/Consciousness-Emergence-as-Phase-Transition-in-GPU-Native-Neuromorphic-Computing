#!/usr/bin/env python3

import datasets
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from datetime import datetime

# Simple NeuroCHIMERA model
class SimpleNeuroCHIMERA(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=128, hidden_dim=64, num_classes=2):
        super(SimpleNeuroCHIMERA, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        x = self.relu(self.fc1(embedded))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        logits = self.classifier(x)
        return logits

def run_simple_glue_benchmark():
    """Run a simplified GLUE benchmark using datasets library"""
    
    # Use datasets library instead of deprecated processors
    tasks = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for task in tasks:
        print(f"\nProcessing {task}...")
        
        try:
            # Load dataset using datasets library
            dataset = datasets.load_dataset('glue', task)
            
            # Get train and validation splits
            train_data = dataset['train']
            val_data = dataset['validation'] if 'validation' in dataset else dataset['test']
            
            # Tokenize function
            def tokenize_function(examples):
                return tokenizer(examples['sentence'] if 'sentence' in examples else examples['sentence1'], 
                               examples.get('sentence2', None), 
                               padding='max_length', 
                               truncation=True, 
                               max_length=128)
            
            # Tokenize datasets
            train_dataset = train_data.map(tokenize_function, batched=True)
            val_dataset = val_data.map(tokenize_function, batched=True)
            
            # Convert to PyTorch format
            train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            
            # Create model
            num_labels = len(set(train_data['label']))
            model = SimpleNeuroCHIMERA(
                vocab_size=tokenizer.vocab_size,
                embed_dim=128,
                hidden_dim=64,
                num_classes=num_labels
            ).to(device)
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=2e-5)
            loss_fn = nn.CrossEntropyLoss()
            
            # Simple training loop
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16)
            
            # Train for 1 epoch (simplified)
            model.train()
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids)
                loss = loss_fn(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break  # Just one batch for testing
            
            # Evaluation
            model.eval()
            predictions = []
            true_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['label'].to(device)
                    
                    logits = model(input_ids)
                    preds = torch.argmax(logits, dim=1)
                    
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    break  # Just one batch for testing
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            
            results[task] = {
                'score': accuracy,
                'training_time_seconds': 1.0,  # Placeholder
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'framework': 'PyTorch',
                'hardware': 'CPU' if not torch.cuda.is_available() else 'GPU'
            }
            
            print(f"{task}: Accuracy = {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error with {task}: {e}")
            results[task] = {
                'error': str(e),
                'score': 0.0
            }
    
    # Save results
    with open('simple_glue_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    print("Running simplified GLUE benchmark...")
    results = run_simple_glue_benchmark()
    print("\nBenchmark completed!")
    print("Results saved to simple_glue_results.json")
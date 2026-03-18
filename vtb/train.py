# vtb/train.py
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
from .utils import device
import pickle

# vtb/train.py

import torch
import torch.utils.data as data
import numpy as np
import pickle

class PatientDataset(data.Dataset):
    def __init__(self, cache_paths):
        self.cache_paths = cache_paths
        self.sample_indices = []
        self.path_indices = []
        
        # Build index of all samples
        for i, cache_path in enumerate(cache_paths):
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                num_samples = len(cache_data['inputs'])
                self.sample_indices.extend(range(num_samples))
                self.path_indices.extend([i] * num_samples)
        
        self.length = len(self.sample_indices)
        
        # Add a dictionary to cache loaded patient data
        # Key: index of cache_path (i), Value: loaded cache_data
        self._patient_cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path_idx = self.path_indices[idx]
        sample_idx = self.sample_indices[idx]
        
        # Check if this patient's data is already in cache
        if path_idx not in self._patient_cache:
            with open(self.cache_paths[path_idx], 'rb') as f:
                self._patient_cache[path_idx] = pickle.load(f)
        
        cache_data = self._patient_cache[path_idx]
        
        inputs = cache_data['inputs'][sample_idx]
        targets = cache_data['targets'][sample_idx]
        labels = cache_data['labels'][sample_idx]
        
        return (
            torch.tensor(inputs, dtype=torch.float),
            torch.tensor(targets, dtype=torch.float),
            torch.tensor(labels, dtype=torch.long)
        )

    def __del__(self):
        self._patient_cache.clear()

    def clear_cache(self):
        """
        Manually clear the loaded patient data cache.
        Call after each epoch to free memory.
        """
        self._patient_cache.clear()
  





# >>> paste from original: def train_NN <<<
def train_NN(model, train_data, test_data, batch_size=50, num_epochs=100, lr=1e-3, l2=0, 
             model_path=None, patience=10, min_delta=1e-5):
    # Create datasets and dataloaders
    train_dataset = PatientDataset(train_data)
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True, num_workers=1)
    
    # Similarly for test data if available
    test_iter = None
    if test_data and len(test_data) > 0:
        test_dataset = PatientDataset(test_data)
        test_iter = data.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses, test_losses = [], []

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        batch_counter = 0 
        for X, y, labels in train_iter: 
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            
            y_hat = model(X, labels)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
            batch_counter += 1
            if batch_counter % 50 == 0:
                train_dataset.clear_cache()     
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        test_loss = None
        if test_iter is not None:
            model.eval()
            test_loss = 0.0
            batch_counter = 0  
            with torch.no_grad():
                for X, y, labels in test_iter: 
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    y_hat = model(X, labels) 
                    loss = loss_fn(y_hat, y)
                    test_loss += loss.item() * X.size(0)
                    batch_counter += 1
                    if batch_counter % 50 == 0:
                        test_dataset.clear_cache()  
                        
            test_loss /= len(test_dataset)
            test_losses.append(test_loss)
            if test_loss < best_loss - min_delta:
                best_loss = test_loss
                epochs_no_improve = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            if train_loss < best_loss - min_delta:
                best_loss = train_loss
                epochs_no_improve = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        epoch_time = time.time() - start_time
        if test_loss is not None:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_time:.2f}s | "
                  f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Best Test Loss: {best_loss:.6f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.6f}")
 
        train_dataset.clear_cache()
        
        
        test_dataset.clear_cache()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if model_path and best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_loss': best_loss
        }, model_path)
        print(f"Best model saved to {model_path}")
    return model, train_losses, test_losses


def save_model(model, path, train_losses, test_losses, best_loss):
    torch.save({
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_loss': best_loss
    }, path)
    print(f"Model saved to {path}")

def load_model(model, path):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded from {path}")
    return model, checkpoint.get('train_losses', []), checkpoint.get('test_losses', []), checkpoint.get('best_loss', None)


def fine_tune_for_patient(pretrained_model, patient_data, 
                          batch_size=16, num_epochs=200, lr=1e-5, 
                          l2=1e-4, patience=25, min_delta=1e-4, output_dir=None):
    import matplotlib.pyplot as plt
    import os
    inputs, targets, labels = patient_data 
    
    fine_tuned_model = type(pretrained_model)(**pretrained_model.init_args)
    fine_tuned_model.load_state_dict(pretrained_model.state_dict())
    fine_tuned_model.to(device)

    split_idx = int(0.9 * len(inputs))
    train_inputs, val_inputs  = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    train_labels, val_labels  = labels[:split_idx], labels[split_idx:]

    train_inputs = torch.tensor(train_inputs, dtype=torch.float)
    train_targets = torch.tensor(train_targets, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    val_inputs   = torch.tensor(val_inputs, dtype=torch.float)
    val_targets  = torch.tensor(val_targets, dtype=torch.float)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    
    train_dataset = data.TensorDataset(train_inputs, train_targets, train_labels) 
    val_dataset   = data.TensorDataset(val_inputs, val_targets, val_labels)       
    
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_iter   = data.DataLoader(val_dataset, batch_size, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=lr, weight_decay=l2)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses, val_losses = [], []

    print(f"Fine-tuning on patient data for up to {num_epochs} epochs...")
    for epoch in range(num_epochs):
        fine_tuned_model.train()
        train_loss = 0.0
        for X, y, batch_labels in train_iter: 
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            y_hat = fine_tuned_model(X, batch_labels) 
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        fine_tuned_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y, batch_labels in val_iter: 
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                y_hat = fine_tuned_model(X, batch_labels) 
                loss = loss_fn(y_hat, y)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.cpu() for k, v in fine_tuned_model.state_dict().items()}
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} during fine-tuning")
            break

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Fine-tune Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if best_model_state is not None:
        fine_tuned_model.load_state_dict(best_model_state)


    return fine_tuned_model, train_losses, val_losses

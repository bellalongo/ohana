import argparse
import yaml
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys
import numpy as np
from sklearn.metrics import confusion_matrix

# Add the parent directory to the path to allow for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ohana.models.crnn_attention import CRNNAttention
from ohana.training.dataset import OhanaDataset

def train(model, device, train_loader, optimizer, loss_fn, epoch):
    """Training loop for one epoch."""
    model.train()
    batch_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")
    for batch_idx, batch in enumerate(pbar):
        data, target = batch['patch'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        
        loss_item = loss.item()
        batch_losses.append(loss_item)
        pbar.set_postfix(loss=f"{loss_item:.4f}")
        
    avg_loss = np.mean(batch_losses)
    print(f"Epoch {epoch} [TRAIN] Average Loss: {avg_loss:.4f}")
    return batch_losses

def validate(model, device, val_loader, loss_fn, class_map):
    """Validation loop that also returns predictions for confusion matrix."""
    model.eval()
    batch_losses = []
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[VALIDATE]")
        for batch in pbar:
            data, target = batch['patch'].to(device), batch['label'].to(device)
            logits, _ = model(data)
            
            loss = loss_fn(logits, target)
            batch_losses.append(loss.item())
            
            _, predicted = torch.max(logits.data, 1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            pbar.set_postfix(acc=f"{(100 * np.sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)):.2f}%")

    avg_loss = np.mean(batch_losses)
    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)
    print(f"[VALIDATE] Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # --- Confusion Matrix Calculation ---
    cm = confusion_matrix(all_targets, all_predictions, labels=list(class_map.values()))
    class_names = list(class_map.keys())
    print("Confusion Matrix (Predicted vs. True):")
    header = f"{'':<12} | " + ' '.join([f'{name:<12}' for name in class_names])
    print(header)
    print('-' * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<12} | " + ' '.join([f'{val:<12}' for val in row])
        print(row_str)

    return accuracy, batch_losses

def main():
    parser = argparse.ArgumentParser(description="Train a CRNN+Attention model on ohana data.")
    parser.add_argument('--config', type=str, required=True, help='Path to the creator config YAML file.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation.')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Directory to save trained models.')
    args = parser.parse_args()

    # --- Setup ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading dataset...")
    data_location = config.get('data_dir', config.get('output_dir'))
    if not data_location:
        raise KeyError("Config file must contain either 'data_dir' or 'output_dir' key to specify data location.")

    full_dataset = OhanaDataset(
        data_dir=data_location,
        patch_size=config['patch_size']
    )
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Please generate data first using create_training_set.py")
        return

    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Using batch size: {args.batch_size}")

    # --- Model, Loss, Optimizer ---
    model = CRNNAttention(num_classes=len(full_dataset.class_map))
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    class_weights = full_dataset.get_class_weights().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training & Validation Loop ---
    best_accuracy = 0.0
    history = {
        "train_loss_per_batch": [],
        "val_loss_per_batch": [],
        "val_accuracy_per_epoch": []
    }

    for epoch in range(1, args.epochs + 1):
        train_batch_losses = train(model, device, train_loader, optimizer, loss_fn, epoch)
        accuracy, val_batch_losses = validate(model, device, val_loader, loss_fn, full_dataset.class_map)
        
        history["train_loss_per_batch"].extend(train_batch_losses)
        history["val_loss_per_batch"].extend(val_batch_losses)
        history["val_accuracy_per_epoch"].append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_save_path = os.path.join(args.output_dir, "best_model.pth")
            print(f"New best accuracy! Saving model to {model_save_path}")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            
    print("Training finished.")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")

    # --- Save Training History ---
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

if __name__ == '__main__':
    main()

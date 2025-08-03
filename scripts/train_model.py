import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys

# Add the parent directory to the path to allow for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ohana.models.crnn_attention import CRNNAttention
from ohana.training.dataset import OhanaDataset

def train(model, device, train_loader, optimizer, loss_fn, epoch):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")
    for batch_idx, batch in enumerate(pbar):
        data, target = batch['patch'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} [TRAIN] Average Loss: {avg_loss:.4f}")

def validate(model, device, val_loader, loss_fn):
    """Validation loop."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[VALIDATE]")
        for batch in pbar:
            data, target = batch['patch'].to(device), batch['label'].to(device)
            logits, _ = model(data)
            val_loss += loss_fn(logits, target).item()
            
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix(acc=f"{(100 * correct / total):.2f}%")

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"[VALIDATE] Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train a CRNN+Attention model on ohana data.")
    parser.add_argument('--config', type=str, required=True, help='Path to the creator config YAML file.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
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
    full_dataset = OhanaDataset(
        data_dir=config['output_dir'],
        patch_size=config['patch_size']
    )
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Please generate data first using create_training_set.py")
        return

    # Split dataset
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- Model, Loss, Optimizer ---
    model = CRNNAttention(num_classes=4) # Don't send to device yet

    # Add this block to wrap the model for multi-GPU data parallelism
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device) # Now send the (potentially wrapped) model to the device
    print(model)
    
    # Use class weights to handle imbalance
    class_weights = full_dataset.get_class_weights().to(device)
    print(f"Using class weights: {class_weights}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training & Validation Loop ---
    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, loss_fn, epoch)
        accuracy = validate(model, device, val_loader, loss_fn)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_save_path = os.path.join(args.output_dir, "best_model.pth")
            print(f"New best accuracy! Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
            
    print("Training finished.")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()


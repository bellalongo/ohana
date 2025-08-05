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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ohana.models.unet_3d import UNet3D
from ohana.training.dataset import OhanaDataset
from torch.cuda.amp import GradScaler, autocast

def train(model, device, train_loader, optimizer, loss_fn, epoch, scaler):
    model.train()
    batch_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")
    for batch in pbar:
        data, target_mask = batch['patch'].to(device), batch['mask'].to(device)
        
        # --- NEW: More robust min-max normalization ---
        # This scales each patch in the batch to the range [0, 1]
        b, c, t, h, w = data.shape
        data_flat = data.view(b, -1)
        min_val = data_flat.min(dim=1, keepdim=True)[0]
        max_val = data_flat.max(dim=1, keepdim=True)[0]
        # Add epsilon to the denominator to prevent division by zero for flat patches
        data_flat = (data_flat - min_val) / (max_val - min_val + 1e-6)
        data = data_flat.view(b, c, t, h, w)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            output_logits = model(data)
            T_out = output_logits.shape[2]
            central_frame_logits = output_logits[:, :, T_out // 2, :, :]
            loss = loss_fn(central_frame_logits, target_mask)

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        loss_item = loss.item()
        if np.isnan(loss_item):
            continue
            
        batch_losses.append(loss_item)
        pbar.set_postfix(loss=f"{loss_item:.4f}")
        
    avg_loss = np.mean(batch_losses) if batch_losses else float('nan')
    print(f"Epoch {epoch} [TRAIN] Average Loss: {avg_loss:.4f}")

def validate(model, device, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    total_anomaly_pixels = 0
    correct_anomaly_pixels = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[VALIDATE]")
        for batch in pbar:
            data, target_mask = batch['patch'].to(device), batch['mask'].to(device)
            
            # --- NEW: Add min-max normalization here as well ---
            b, c, t, h, w = data.shape
            data_flat = data.view(b, -1)
            min_val = data_flat.min(dim=1, keepdim=True)[0]
            max_val = data_flat.max(dim=1, keepdim=True)[0]
            data_flat = (data_flat - min_val) / (max_val - min_val + 1e-6)
            data = data_flat.view(b, c, t, h, w)
            
            with autocast():
                output_logits = model(data)
                T_out = output_logits.shape[2]
                central_frame_logits = output_logits[:, :, T_out // 2, :, :]
                loss = loss_fn(central_frame_logits, target_mask)
                
            total_loss += loss.item()
            preds = torch.argmax(central_frame_logits, dim=1)
            anomaly_mask = target_mask > 0
            total_anomaly_pixels += anomaly_mask.sum().item()
            correct_anomaly_pixels += (preds[anomaly_mask] == target_mask[anomaly_mask]).sum().item()

    avg_loss = total_loss / len(val_loader)
    anomaly_accuracy = (correct_anomaly_pixels / total_anomaly_pixels * 100) if total_anomaly_pixels > 0 else 0
    
    print(f"[VALIDATE] Average Loss: {avg_loss:.4f}, Anomaly Pixel Accuracy: {anomaly_accuracy:.2f}%")
    return avg_loss, anomaly_accuracy

def main():
    parser = argparse.ArgumentParser(description="Train a 3D U-Net model on ohana data.")
    parser.add_argument('--config', type=str, required=True, help='Path to the creator config YAML file.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for 3D training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data for validation.')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Directory to save models.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_location = config.get('output_dir')
    full_dataset = OhanaDataset(data_dir=data_location, patch_size=config['patch_size'])
    
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    model = UNet3D(n_channels=1, n_classes=len(full_dataset.class_map))
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scaler = GradScaler()

    best_val_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, loss_fn, epoch, scaler)
        val_loss, val_accuracy = validate(model, device, val_loader, loss_fn)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(args.output_dir, "best_model_unet3d.pth")
            print(f"New best validation accuracy: {best_val_accuracy:.2f}%. Saving model...")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()
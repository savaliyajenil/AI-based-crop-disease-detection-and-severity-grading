import os
import argparse
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
# Add src to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.samt_seg import SAMTSeg, SeverityConsistencyLoss

# ==============================================================================
# CONFIG & METRICS
# ==============================================================================
CHECKPOINT_DIR = Path(r"D:\pdeu_kaam\checkpoints")
RESULTS_DIR = Path(r"D:\pdeu_kaam\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = RESULTS_DIR / "training_log.csv"
PLOT_FILE = RESULTS_DIR / "training_curves.png"

def compute_metrics(pred_logits, true_masks, num_classes):
    """
    Computes Pixel Accuracy, mIoU, and Mean Dice Coefficient.
    pred_logits: (B, C, H, W)
    true_masks: (B, H, W)
    """
    # Get class predictions
    preds = torch.argmax(pred_logits, dim=1) # (B, H, W)
    
    # 1. Pixel Accuracy
    correct = (preds == true_masks).sum().item()
    total = true_masks.numel()
    acc = correct / total
    
    # 2. mIoU & Dice
    iou_per_class = []
    dice_per_class = []
    
    for c in range(num_classes):
        pred_c = (preds == c)
        true_c = (true_masks == c)
        
        intersection = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        
        if union == 0:
            # If class is not present in both true and pred, ignore it in mean
            continue
            
        iou = intersection / union
        iou_per_class.append(iou)
        
        # Dice = 2 * Intersection / (Pred Area + True Area)
        area_sum = pred_c.sum().item() + true_c.sum().item()
        dice = (2.0 * intersection) / area_sum if area_sum > 0 else 0
        dice_per_class.append(dice)
        
    mIoU = sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0
    mDice = sum(dice_per_class) / len(dice_per_class) if dice_per_class else 0
    
    return acc, mIoU, mDice

def save_checkpoint(model, optimizer, scaler, epoch, metrics, filename):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CHECKPOINT_DIR / filename
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, scaler, filename):
    filepath = CHECKPOINT_DIR / filename
    if not filepath.exists():
        return 0, {}
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'] + 1, checkpoint.get('metrics', {})

def log_to_csv(epoch, seg_loss, sev_loss, cons_loss, total_loss, acc, miou, mdice):
    file_exists = CSV_FILE.exists()
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Seg_Loss', 'Sev_Loss', 'Consistency_Loss', 'Total_Loss', 'Pixel_Acc', 'mIoU', 'mDice'])
        writer.writerow([epoch, f"{seg_loss:.4f}", f"{sev_loss:.4f}", f"{cons_loss:.4f}", f"{total_loss:.4f}", f"{acc:.4f}", f"{miou:.4f}", f"{mdice:.4f}"])

def plot_training_curves():
    if not CSV_FILE.exists(): return
    epochs, t_losses, mious, mdices, accs = [], [], [], [], []
    with open(CSV_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['Epoch']))
            t_losses.append(float(row['Total_Loss']))
            mious.append(float(row['mIoU']))
            mdices.append(float(row['mDice']))
            accs.append(float(row['Pixel_Acc']))
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Loss
    ax1.plot(epochs, t_losses, label='Total Loss', color='red', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Metrics
    ax2.plot(epochs, mious, label='mIoU', color='blue')
    ax2.plot(epochs, mdices, label='mDice', color='green')
    ax2.plot(epochs, accs, label='Pixel Acc', color='purple', linestyle='--')
    ax2.set_title('Segmentation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score (0 to 1)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()

def train(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    model = SAMTSeg(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    
    seg_criterion = nn.CrossEntropyLoss()
    sev_criterion = nn.MSELoss()
    consist_criterion = SeverityConsistencyLoss(background_class_idx=0)
    
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, scaler, "latest_checkpoint.pth")
    else:
        if CSV_FILE.exists(): CSV_FILE.unlink()
    
    print("\n[INFO] Starting Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_seg_loss, ep_sev_loss, ep_cons_loss, ep_total_loss = 0.0, 0.0, 0.0, 0.0
        ep_acc, ep_miou, ep_mdice = 0.0, 0.0, 0.0
        
        num_batches = 10
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx in pbar:
            # Generate dummy data (Replace with actual DataLoader)
            images = torch.randn(args.batch_size, 3, 256, 256).to(device, non_blocking=True)
            true_masks = torch.randint(0, args.num_classes, (args.batch_size, 256, 256)).to(device, non_blocking=True)
            true_severity = torch.rand(args.batch_size, 1).to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                pred_masks, pred_severity = model(images)
                loss_seg = seg_criterion(pred_masks, true_masks)
                loss_sev = sev_criterion(pred_severity, true_severity)
                loss_consist = consist_criterion(pred_masks, pred_severity)
                total_loss = loss_seg + (args.lambda_sev * loss_sev) + (args.lambda_cons * loss_consist)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Compute metrics for this batch
            with torch.no_grad():
                acc, miou, mdice = compute_metrics(pred_masks, true_masks, args.num_classes)
            
            ep_seg_loss += loss_seg.item()
            ep_sev_loss += loss_sev.item()
            ep_cons_loss += loss_consist.item()
            ep_total_loss += total_loss.item()
            ep_acc += acc
            ep_miou += miou
            ep_mdice += mdice
            
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'mIoU': f"{miou:.4f}"})
        
        # Epoch averages
        avg_tot = ep_total_loss / num_batches
        avg_acc = ep_acc / num_batches
        avg_miou = ep_miou / num_batches
        avg_mdice = ep_mdice / num_batches
        
        print(f"Epoch {epoch+1} | Loss: {avg_tot:.4f} | mIoU: {avg_miou:.4f} | mDice: {avg_mdice:.4f} | Acc: {avg_acc:.4f}")
        
        log_to_csv(epoch+1, ep_seg_loss/num_batches, ep_sev_loss/num_batches, 
                   ep_cons_loss/num_batches, avg_tot, avg_acc, avg_miou, avg_mdice)
        plot_training_curves()
        
        metrics = {'val_loss': avg_tot, 'val_miou': avg_miou}
        save_checkpoint(model, optimizer, scaler, epoch, metrics, "latest_checkpoint.pth")
            
    print(f"\n[INFO] Training Complete. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lambda_sev', type=float, default=1.0)
    parser.add_argument('--lambda_cons', type=float, default=0.5)
    args = parser.parse_args()
    train(args)

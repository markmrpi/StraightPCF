import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.laz_hag_dataset import LazHAGDataset
from models.hag_predictor import HAGPredictor


def train():
    parser = argparse.ArgumentParser(description="Train HAG predictor")
    parser.add_argument("--data-root", default="./data", help="Root directory containing train/val/test folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--out-dir", default="./logs/hag", help="Where to store checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LazHAGDataset(args.data_root, split="train", with_hag=True)
    val_ds = LazHAGDataset(args.data_root, split="val", with_hag=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = HAGPredictor(hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            xyz = batch["xyz"].to(device)
            hag = batch["hag"].to(device)
            pred = model(xyz).squeeze()
            loss = criterion(pred, hag.squeeze())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                xyz = batch["xyz"].to(device)
                hag = batch["hag"].to(device)
                pred = model(xyz).squeeze()
                val_losses.append(criterion(pred, hag.squeeze()).item())
        print(f"Epoch {epoch+1}/{args.epochs}: val_loss={np.mean(val_losses):.6f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "hag_predictor.pth"))


if __name__ == "__main__":
    train()

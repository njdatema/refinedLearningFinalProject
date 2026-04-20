import os
import json
import torch
import torch.nn as nn
from torchvision import transforms

from model import CSRNet
import dataset   # or your renamed dataset file

def validate(val_list, model, device):
    print('begin evaluation')

    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(
            val_list,
            shuffle=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]),
            train=False
        ),
        batch_size=1
    )

    model.eval()
    mae = 0.0
    mse = 0.0

    with torch.no_grad():
        for i, (img, target) in enumerate(val_loader):
            img = img.to(device)
            output = model(img)

            target = target.float().to(device)
            if target.dim() == 2:
                target = target.unsqueeze(0).unsqueeze(0)
            elif target.dim() == 3:
                target = target.unsqueeze(1)

            pred_count = output.sum().item()
            gt_count = target.sum().item()

            err = pred_count - gt_count
            mae += abs(err)
            mse += err * err

            print(f"[{i+1}/{len(val_loader)}] pred={pred_count:.2f}, gt={gt_count:.2f}, abs_err={abs(err):.2f}")

    mae /= len(val_loader)
    rmse = (mse / len(val_loader)) ** 0.5

    print(f"\nMAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open("val.json", "r") as f:
        val_list = json.load(f)

    model = CSRNet().to(device)

    checkpoint = torch.load("weights.pth", map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint format")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state_dict format")

    validate(val_list, model, device)

if __name__ == "__main__":
    main()
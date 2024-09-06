import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.amp
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

def define_transforms():
    train_transform = A.Compose([
        A.Resize(128, 128),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0] * 12, std=[1.0] * 12),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=[0.0] * 12, std=[1.0] * 12),
        ToTensorV2(),
    ])
    return train_transform, val_transforms


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().to(device=device)  # Remove unsqueeze operation
        print(f"data shape: {data.shape}, targets shape: {targets.shape}")

        optimizer.zero_grad()

        if device == 'cuda':
            with torch.amp.autocast(device_type=device):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            loss.backward()
            optimizer.step()

        loop.set_postfix(loss=loss.item())


def main():
    LOAD_MODEL = False  # Set this to True if you want to load a previously trained model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {DEVICE}")

    model = UNET(in_channels=12, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler() if DEVICE == 'cuda' else None

    train_transform, val_transforms = define_transforms()
    train_loader, val_loader = get_loaders(
        "data/train_images/", "data/train_masks/",
        "data/val_images/", "data/val_masks/",
        16, train_transform, val_transforms,
        2, True)

    if LOAD_MODEL and os.path.isfile("my_checkpoint.pth.tar"):
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    for epoch in range(3):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        check_accuracy(val_loader, model, device=DEVICE)
        if DEVICE == 'cuda':
            save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()})
        save_predictions_as_imgs(val_loader, model, folder="saved_images", device=DEVICE)

if __name__ == "__main__":
    main()

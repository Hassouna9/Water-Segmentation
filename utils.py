import torch
import matplotlib.pyplot as plt
import os
import torchvision
from dataset import SatelliteDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SatelliteDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        target_transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SatelliteDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_tensor_as_image(tensor, folder, base_filename):
    """ Saves each tensor in a batch by selecting up to the first three channels if more than 4 channels are present. """
    if tensor.dim() == 4:  # Check for batch dimension
        for i, single_tensor in enumerate(tensor):
            _save_single_image(single_tensor, folder, f"{base_filename}_{i}.png")
    else:
        _save_single_image(tensor, folder, base_filename)

def _save_single_image(tensor, folder, filename):
    """ Helper function to save a single image tensor. """
    plt.figure()
    if tensor.shape[0] == 1:  # Grayscale
        plt.imshow(tensor.squeeze(0).cpu().numpy(), cmap='gray')
    elif tensor.shape[0] <= 4:  # Standard RGB or RGBA
        plt.imshow(tensor.permute(1, 2, 0).cpu().numpy())
    else:  # More than 4 channels, visualize the first three as RGB
        plt.imshow(tensor[:3].permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # Thresholding to binary values
            preds = (preds * 255).byte()  # Scale to 0-255 and convert to uint8

        # Save predictions using custom function
        save_tensor_as_image(preds.float() / 255, folder, f"pred_{idx}")  # Rescale to [0.0, 1.0] for saving

        # Adjust and save ground truth images if necessary
        if y.dim() == 4:
            y = y.squeeze(0)  # Reduce it to [channels, height, width]
        save_tensor_as_image(y.float() / 255, folder, f"y_{idx}")  # Ensure y is also properly scaled

    model.train()






import torch
import matplotlib.pyplot as plt
import os
import numpy as np
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
def calculate_iou(preds, targets, threshold=0.3):
    """ Calculate Intersection over Union (IoU) for water class (class = 1). """
    preds = (preds > threshold).float()  # Convert predictions to binary (water class vs background)
    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection
    iou = intersection / (union + 1e-8)  # Add epsilon to avoid division by zero
    return iou.item()

def calculate_precision(preds, targets, threshold=0.5):
    """ Calculate Precision for water class (class = 1). """
    preds = (preds > threshold).float()
    true_positives = (preds * targets).sum()
    predicted_positives = preds.sum()
    precision = true_positives / (predicted_positives + 1e-8)
    return precision.item()

def calculate_recall(preds, targets, threshold=0.5):
    """ Calculate Recall for water class (class = 1). """
    preds = (preds > threshold).float()
    true_positives = (preds * targets).sum()
    actual_positives = targets.sum()
    recall = true_positives / (actual_positives + 1e-8)
    return recall.item()

def calculate_f1(precision, recall):
    """ Calculate F1-score from precision and recall. """
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
def visualize_rgb(image_tensor, folder="rgb_visualization", file_prefix="image", red_band=3, green_band=2, blue_band=1):
    """ Visualize the multispectral image as an RGB image by combining specified bands. """
    os.makedirs(folder, exist_ok=True)  # Ensure directory exists

    # Extract the RGB bands
    red = image_tensor[red_band, :, :].cpu().numpy()
    green = image_tensor[green_band, :, :].cpu().numpy()
    blue = image_tensor[blue_band, :, :].cpu().numpy()

    # Normalize the bands to the [0, 1] range for visualization (if necessary)
    red = (red - red.min()) / (red.max() - red.min())
    green = (green - green.min()) / (green.max() - green.min())
    blue = (blue - blue.min()) / (blue.max() - blue.min())

    # Stack the channels to create an RGB image
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Plot and save the RGB image
    plt.figure()
    plt.imshow(rgb_image)
    plt.title("RGB Visualization")
    plt.axis('off')  # Turn off axis labels for clean visualization

    file_path = os.path.join(folder, f"{file_prefix}_rgb.png")
    plt.savefig(file_path)
    plt.close()

    print(f"RGB image visualized and saved at {file_path}")


def check_accuracy(loader, model, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    count = 0  # To track the number of batches for averaging

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)  # Ensure mask has the same shape as the output

            # Get model predictions
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )
            # Compute metrics for each batch
            iou = calculate_iou(preds, y)
            precision = calculate_precision(preds, y)
            recall = calculate_recall(preds, y)
            f1 = calculate_f1(precision, recall)

            # Update totals for averaging later
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1

    # Calculate average metrics across all batches
    avg_iou = total_iou / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_f1 = total_f1 / count
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
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



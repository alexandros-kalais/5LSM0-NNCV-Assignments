import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import numpy as np
from PIL import Image
from unet import UNet
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}

CITYSCAPES_PALETTE = np.array([
    (128, 64,128), (244, 35,232), (70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    (70,130,180), (220, 20, 60), (255,  0,  0), (0,  0,142), (0,  0, 70),
    (0, 60,100), (0, 80,100), (0,  0,230), (119, 11, 32)
], dtype=np.uint8)

def convert_to_train_ids(label):
    label = np.array(label)
    label_copy = 255 * np.ones_like(label, dtype=np.uint8)
    for k, v in CITYSCAPES_ID_TO_TRAINID.items():
        label_copy[label == k] = v
    return Image.fromarray(label_copy)

def decode_segmap(label_mask):
    h, w = label_mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx in range(19):
        color_image[label_mask == class_idx] = CITYSCAPES_PALETTE[class_idx]
    return color_image

class CityscapesTrainIDWrapper(Cityscapes):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        label = convert_to_train_ids(label)
        return image, label

def evaluate_model(model, dataloader):
    model.eval()
    iou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(device)
    acc_metric = MulticlassAccuracy(num_classes=19, ignore_index=255).to(device)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.long().squeeze(1)
            logits = model(imgs)
            logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = logits.argmax(dim=1)
            iou_metric.update(preds, labels)
            acc_metric.update(preds, labels)

    return {
        "IoU": iou_metric.compute().item(),
        "Accuracy": acc_metric.compute().item()
    }

def main():
    transform = Compose([
        ToImage(), Resize((256, 256)), ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.5,) * 3, std=(0.5,) * 3)
    ])
    resize_label = Resize((256, 256), interpolation=Image.NEAREST)

    model_configs = [
        ("Stage 0", "/home/scur1343/5LSM0-NNCV-Assignments/Final assignment/checkpoints/dice_loss_no_aug/best_model-epoch=0099-val_loss=0.46753984689712524.pth"),
        ("Stage 1", "/home/scur1343/5LSM0-NNCV-Assignments/Final assignment/checkpoints/stage_1_weather_conditions/best_model-epoch=0094-val_loss=0.43855899572372437.pth"),
        ("Stage 2", "/home/scur1343/5LSM0-NNCV-Assignments/Final assignment/checkpoints/stage_2_weather_conditions/best_model-epoch=0094-val_loss=0.47626787424087524.pth"),
        ("Combined Stages", "/home/scur1343/5LSM0-NNCV-Assignments/Final assignment/checkpoints/combined_stages/best_model-epoch=0091-val_loss=0.4116141200065613.pth")
    ]

    models = {}
    for name, path in model_configs:
        model = UNet(in_channels=3, n_classes=19).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models[name] = model

    dataset = CityscapesTrainIDWrapper(
        "./data/cityscapes", split="val", mode="fine", target_type="semantic", transforms=transform
    )
    dataset = wrap_dataset_for_transforms_v2(dataset)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    indices = [0, 20, 40]
    samples = [dataset[i] for i in indices]

    # --- Plot predictions ---
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
    col_titles = ["Ground Truth"] + list(models.keys())

    for row, (img, label) in enumerate(samples):
        img_input = transform(img).unsqueeze(0).to(device)
        label = resize_label(label)
        gt_np = np.array(label).squeeze().astype(np.uint8)
        axes[row, 0].imshow(decode_segmap(gt_np))
        axes[row, 0].axis("off")
        for col, (model_name, model) in enumerate(models.items(), start=1):
            with torch.no_grad():
                output = model(img_input)
                pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            axes[row, col].imshow(decode_segmap(pred))
            axes[row, col].axis("off")

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.savefig("qualitative_comparison_cityscapes.png", dpi=200)
    plt.show()

    # --- Evaluate and print metrics ---
    print(f"\n{'Model':<20} | {'mIoU':>6} | {'Accuracy':>8}")
    print("-" * 40)
    for name, model in models.items():
        metrics = evaluate_model(model, val_loader)
        print(f"{name:<20} | {metrics['IoU']:.4f} | {metrics['Accuracy']:.4f}")

if __name__ == "__main__":
    main()

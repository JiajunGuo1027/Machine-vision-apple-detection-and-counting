import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
import cv2
from PIL import Image
import numpy as np

# Dataset definition
class AppleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir=None, transforms=None, is_test=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.is_test = is_test
        self.imgs = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if not self.is_test:
            mask_path = os.path.join(self.mask_dir, self.imgs[idx].replace('.jpg', '.png'))
            mask = Image.open(mask_path).convert("L")
            boxes, labels = self.parse_mask(mask)

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
        else:
            target = {}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def parse_mask(self, mask):
        mask = np.array(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        labels = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
        return boxes, labels

    def __len__(self):
        return len(self.imgs)

#Model definition function
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

# Function to save the model
def save_model(model, optimizer, epoch, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)  # 创建保存文件夹
    save_path = os.path.join(save_dir, f"fasterrcnn_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, save_path)
    print(f"Model saved to {save_path}")

# Function to load the model
def load_model(save_path, num_classes, device):
    model = get_model(num_classes)
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Model loaded from {save_path}, starting from epoch {epoch}")
    return model, optimizer, epoch

#Data loading
train_dataset = AppleDataset(
    img_dir="C:/Users/31606/Desktop/MV_dataset/detection/train/images",
    mask_dir="C:/Users/31606/Desktop/MV_dataset/detection/train/masks",
    transforms=F.to_tensor
)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

test_dataset = AppleDataset(
    img_dir="C:/Users/31606/Desktop/MV_dataset/detection/test/images",
    is_test=True,
    transforms=F.to_tensor
)
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

#Configure device; use GPU for acceleration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize model and optimizer
num_classes = 2
model = get_model(num_classes)
model.to(device)
optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#  Check whether to load an existing model
resume_training = False
if resume_training:
    model, optimizer, start_epoch = load_model("saved_models/fasterrcnn_epoch_10.pth", num_classes, device)
else:
    start_epoch = 0

# Train the model
num_epochs = 50
for epoch in range(start_epoch, num_epochs):
    print(f"\n[Training] Epoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    lr_scheduler.step()
    print(f"[Training] Epoch {epoch+1} completed. Total loss: {epoch_loss:.4f}")

    # Save the model
    save_model(model, optimizer, epoch+1)

# Test and visualize
def predict_and_visualize(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):
            print(f"Processing batch {idx+1}/{len(data_loader)}...")
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i, (image, output) in enumerate(zip(images, outputs)):
                print(f"Image {i+1}: Detected {len(output['boxes'])} objects.")
                visualize_detection(image.cpu(), output["boxes"].cpu(), output["labels"].cpu())


def visualize_detection(image, boxes, labels):
    """
    Visualize detection results for a single image.
    :param image: Image tensor
    :param boxes: Bounding box tensor
    :param labels: Class label tensor
    """
    # Convert the image tensor to a numpy array
    image = image.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)  # 转换为 uint8 格式
    
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    if len(boxes) == 0:
        print("No objects detected.")
        return
    

    for box, label in zip(boxes, labels):
        
        box = [int(coord) for coord in box.tolist()]
        print(f"Drawing box: {box}, Label: {label.item()}")  # 调试信息

        # Draw bounding box
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Draw label above the bounding box
        cv2.putText(
            image,
            f"Apple {label.item()}",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    # Display detection results
    cv2.imshow("Detection", image)
    cv2.waitKey(500)  # Automatically close the window after 0.5 seconds
    cv2.destroyAllWindows()


print("[Step 6] Testing and visualization...")
predict_and_visualize(model, test_loader, device)

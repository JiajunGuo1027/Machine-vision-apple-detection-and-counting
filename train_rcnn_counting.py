import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import json

#dataset definition
class AppleCountingDataset(Dataset):
    def __init__(self, img_dir, annotations_path, transforms=None, annotation_format="txt"):
        self.img_dir = img_dir
        self.transforms = transforms
        self.annotation_format = annotation_format
        self.annotations = self.load_annotations(annotations_path)  #Load annotation file

    def load_annotations(self, annotations_path):
        """Load annotations based on the file format, supporting TXT and JSON formats"""
        annotations = {}
        if self.annotation_format == "txt":
            with open(annotations_path, 'r') as f:
                lines = f.readlines()[1:]  # Skip the first header line
                for line in lines:
                    parts = line.strip().split(',')
                    img_name = parts[0]
                    count = int(parts[1])
                    annotations[img_name] = count
        elif self.annotation_format == "json":
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
        else:
            raise ValueError("Unsupported annotation format. Use 'txt' or 'json'.")
        return annotations

    def __getitem__(self, idx):
        img_name = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        target = self.annotations[img_name]  #Get the true target count for the image

        if self.transforms:
            img = self.transforms(img)

        return img, torch.tensor(target, dtype=torch.float32)  #Return the image and the true count

    def __len__(self):
        return len(self.annotations)


# Define the counting model using ResNet18 as the feature extractor
class CountingModel(nn.Module):
    def __init__(self):
        super(CountingModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  #Output a scalar value

    def forward(self, x):
        return self.backbone(x)


#Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs, device, save_dir="saved_models"):
    model.train()
    os.makedirs(save_dir, exist_ok=True)  #Ensure the save directory exists

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            #Forward propagation
            outputs = model(images).squeeze()  #Output is a scalar
            loss = criterion(outputs, targets)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        #Save the model
        save_path = os.path.join(save_dir, f"fasterrcnn_counting_epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path)
        print(f"Model saved to {save_path}")  #Print confirmation of the save path

#Test and evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()
    predicted_counts = []
    ground_truth_counts = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.numpy().tolist()
            outputs = model(images).squeeze().cpu().numpy().tolist()

            predicted_counts.extend(outputs)
            ground_truth_counts.extend(targets)

    # Check if there are predictions and true counts
    if len(predicted_counts) == 0 or len(ground_truth_counts) == 0:
        print("Error: No data for evaluation. Please check the test dataset and annotations.")
        return

    #Compute evaluation metrics
    mae = mean_absolute_error(ground_truth_counts, predicted_counts)
    mse = mean_squared_error(ground_truth_counts, predicted_counts)
    r2 = r2_score(ground_truth_counts, predicted_counts)

    print(f"\nEvaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return mae, mse, r2


#Main workflow
def main():
    # Path configuration
    train_img_dir = "C:/Users/31606/Desktop/MV_dataset/counting/train/images"
    train_annotations_path = "C:/Users/31606/Desktop/MV_dataset/counting/train/train_ground_truth.txt"
    test_img_dir = "C:/Users/31606/Desktop/MV_dataset/test_data/counting/images"
    test_annotations_path = "C:/Users/31606/Desktop/MV_dataset/test_data/counting/ground_truth.json"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #Load training and testing datasets
    train_dataset = AppleCountingDataset(train_img_dir, train_annotations_path, transforms=transform, annotation_format="txt")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = AppleCountingDataset(test_img_dir, test_annotations_path, transforms=transform, annotation_format="json")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Print dataset sizes to check for correct loading
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total test samples: {len(test_loader.dataset)}")

    # Initialize the model, loss function, and optimizer
    model = CountingModel().to(device)
    criterion = nn.MSELoss()  # Use Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    print("[Step 1] Training model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Evaluate the model
    print("[Step 2] Evaluating model...")
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()

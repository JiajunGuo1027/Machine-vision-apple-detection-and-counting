import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np

class AppleCountingDataset(Dataset):
    def __init__(self, img_dir, annotations_path, transforms=None, annotation_format="json"):
        self.img_dir = img_dir
        self.transforms = transforms
        self.annotation_format = annotation_format
        self.annotations = self.load_annotations(annotations_path)  # Load annotation file

    def load_annotations(self, annotations_path):
        """Load JSON format annotation file
        """
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        return annotations

    def __getitem__(self, idx):
        img_name = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        target = int(self.annotations[img_name])  # Ensure the target is of integer type

        if self.transforms:
            img = self.transforms(img)

        return img, torch.tensor(target, dtype=torch.float32)  # Return the image and the true count

    def __len__(self):
        return len(self.annotations)


# Define counting model (same as training)
class CountingModel(torch.nn.Module):
    def __init__(self):
        super(CountingModel, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, 1)  # 输出一个标量值

    def forward(self, x):
        return self.backbone(x)


# Test function
# Test and evaluate the model
def test_model(model, dataloader, device):
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

    #Calculate evaluation metrics
    mae = mean_absolute_error(ground_truth_counts, predicted_counts)
    mse = mean_squared_error(ground_truth_counts, predicted_counts)
    r2 = r2_score(ground_truth_counts, predicted_counts)

    # Calculate Mean Accuracy
    mean_accuracy = sum(1 - abs(p - g) / max(g, 1) for p, g in zip(predicted_counts, ground_truth_counts)) / len(ground_truth_counts)

    print(f"\nEvaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")

    # Perform visualization for evaluation
    visualize_evaluation(predicted_counts, ground_truth_counts)

    return mae, mse, r2, mean_accuracy


def visualize_evaluation(predicted_counts, ground_truth_counts):
    # Scatter plot of ground truth and predicted counts for the test set
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth_counts, predicted_counts, alpha=0.7)
    plt.plot([min(ground_truth_counts), max(ground_truth_counts)], [min(ground_truth_counts), max(ground_truth_counts)], 'r--')
    plt.xlabel("Ground Truth Counts")
    plt.ylabel("Predicted Counts")
    plt.title("Scatter Plot of Ground Truth vs. Predicted Counts")
    plt.grid(True)
    plt.show()

    # Compute and display metrics-related histograms
    errors = [p - g for p, g in zip(predicted_counts, ground_truth_counts)]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, alpha=0.7, color='orange')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Errors")
    plt.grid(True)
    plt.show()

    # Show the trend of predicted values and ground truth counts
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth_counts, label="Ground Truth Counts")
    plt.plot(predicted_counts, label="Predicted Counts", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Counts")
    plt.title("Ground Truth vs. Predicted Counts (Line Plot)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Box plot to show the difference between predictions and ground truth
    plt.figure(figsize=(10, 6))
    plt.boxplot([ground_truth_counts, predicted_counts], labels=["Ground Truth", "Predictions"])
    plt.title("Box Plot of Ground Truth and Predictions")
    plt.grid(True)
    plt.show()


def main():
    # Path configuration
    test_img_dir = "C:/Users/31606/Desktop/MV_dataset/counting/test/images"
    test_annotations_path = "C:/Users/31606/Desktop/MV_dataset/test_data/counting/ground_truth.json"
    model_path = "C:/Users/31606/Desktop/MV/saved_models/fasterrcnn_counting_epoch_50.pth"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    test_dataset = AppleCountingDataset(test_img_dir, test_annotations_path, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load the model
    model = CountingModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {model_path}")

    # test the model
    print("[Step] Testing model...")
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()

import os
import json
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class AppleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.imgs = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        filename = self.imgs[idx]
        return img, filename

    def __len__(self):
        return len(self.imgs)

#Model loading
def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded successfully from {model_path}")
    return model

# Generate bounding boxes from mask files
def mask_to_bboxes(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    bboxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    return bboxes

# Generate COCO-format ground truth
def generate_ground_truth_coco(ground_truth_path, masks_dir, output_path):
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    annotation_id = 1  # Initialize unique ID for annotations
    for img in ground_truth["images"]:
        mask_path = os.path.join(masks_dir, img["filename"])
        if os.path.exists(mask_path):
            bboxes = mask_to_bboxes(mask_path)
            annotations = []
            for x1, y1, x2, y2 in bboxes:
                annotations.append({
                    "id": annotation_id,  # Assign a unique ID to each annotation
                    "image_id": img["id"],
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0
                })
                annotation_id += 1  # Update annotation_id

            if "annotations" not in ground_truth:
                ground_truth["annotations"] = []
            ground_truth["annotations"].extend(annotations)

    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)
    print(f"Generated COCO ground truth saved to {output_path}")


#Main process
def main():

    test_images_dir = "C:/Users/31606/Desktop/MV_dataset/test_data/detection/images"
    masks_dir = "C:/Users/31606/Desktop/MV_dataset/test_data/segmentation/masks"
    ground_truth_path = "C:/Users/31606/Desktop/MV_dataset/test_data/detection/ground_truth.json"
    model_path = "C:/Users/31606/Desktop/MV/saved_models/fasterrcnn_epoch_10.pth"
    output_dir = "C:/Users/31606/Desktop/MV/output"
    generated_gt_path = os.path.join(output_dir, "generated_ground_truth.json")
    predictions_output_path = os.path.join(output_dir, "predictions.json")
    evaluation_output_path = os.path.join(output_dir, "evaluation_results.txt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Generate COCO-format ground truth
    generate_ground_truth_coco(ground_truth_path, masks_dir, generated_gt_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(model_path, num_classes=2)
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = AppleDataset(img_dir=test_images_dir, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    #Load ground_truth.json
    with open(generated_gt_path, "r") as f:
        coco_data = json.load(f)
    image_id_map = {img["filename"]: img["id"] for img in coco_data["images"]}

    predictions = []
    with torch.no_grad():
        for idx, (images, filenames) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for image, output, filename in zip(images, outputs, filenames):
                print(f"Processing {idx + 1}/{len(dataset)}: {filename}")

                # Get the image_id from COCO ground truth
                if filename not in image_id_map:
                    print(f"Image {filename} not found in ground truth, skipping.")
                    continue

                image_id = image_id_map[filename]
                for box, score, label in zip(output["boxes"].cpu().tolist(), output["scores"].cpu().tolist(), output["labels"].cpu().tolist()):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score
                    })

    # Save prediction results
    with open(predictions_output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    # Evaluate predictions
    coco_gt = COCO(generated_gt_path)
    coco_dt = coco_gt.loadRes(predictions_output_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    with open(evaluation_output_path, "w") as f:
        stats = coco_eval.stats
        f.write(f"AP: {stats[0]:.3f}\n")
        f.write(f"AP_0.5: {stats[1]:.3f}\n")
        f.write(f"AP_0.75: {stats[2]:.3f}\n")
        f.write(f"AP_small: {stats[3]:.3f}\n")
        f.write(f"AP_medium: {stats[4]:.3f}\n")
        f.write(f"AP_large: {stats[5]:.3f}\n")

if __name__ == "__main__":
    main()

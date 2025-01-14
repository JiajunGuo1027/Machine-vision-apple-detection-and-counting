import os
import json
import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image

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

#Generate bounding boxes from mask files
def mask_to_bboxes(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    bboxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    return bboxes

# Combine ground_truth.json and mask data to generate COCO format
def generate_coco_from_masks(gt_path, mask_dir, output_path, mapping_path):
    """
    生成 COCO 格式的json文件（既包含images字段，也包含annotations字段）。
    这里会使用 mapping.json 中提供的 filename->ID 映射，保证后续预测时 ID 一致。
    """
    #Read ground_truth.json
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # Read mapping.json
    with open(mapping_path, 'r', encoding='utf-8') as f:
        filename_to_id = json.load(f)

    images = []
    annotations = []
    ann_id = 1

    for img_data in gt_data['images']:
        filename = img_data['filename']
        width, height = img_data['width'], img_data['height']

        #  Retrieve id from mapping.json
        if filename not in filename_to_id:
            #If filename is not found in mapping.json, skip or handle it accordingly
            print(f"Warning: {filename} not in mapping.json, skipping.")
            continue
        img_id = filename_to_id[filename]

        #  Record images information
        images.append({
            'id': img_id,
            'width': width,
            'height': height,
            'file_name': filename
        })

        #Find the corresponding mask for the file
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            bboxes = mask_to_bboxes(mask_path)
            for bbox in bboxes:
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': 1,  
                    'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'iscrowd': 0
                })
                ann_id += 1
        else:
            print(f"Warning: {mask_path} 不存在，跳过该图的mask信息。")

    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': 1, 'name': 'object'}]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

# Evaluation main process
def main():
    # Set paths
    test_images_dir = "C:/Users/31606/Desktop/MV_dataset/test_data/detection/images"
    masks_dir = "C:/Users/31606/Desktop/MV_dataset/test_data/segmentation/masks"
    gt_path = "C:/Users/31606/Desktop/MV_dataset/test_data/detection/ground_truth.json"
    generated_gt_path = "C:/Users/31606/Desktop/MV/generated_coco.json"
    predictions_output_path = "C:/Users/31606/Desktop/MV/predictions.json"
    output_dir = "C:/Users/31606/Desktop/MV/output"
    mapping_path = "C:/Users/31606/Desktop/MV_dataset/test_data/detection/mapping.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Generate COCO format data (including 'images', 'annotations', etc.)
    generate_coco_from_masks(
        gt_path=gt_path,
        mask_dir=masks_dir,
        output_path=generated_gt_path,
        mapping_path=mapping_path
    )

    # Step 2: Load the model
    model_path = "C:/Users/31606/Desktop/MV/saved_models/fasterrcnn_epoch_10.pth"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = fasterrcnn_resnet50_fpn(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Step 3: Load the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = AppleDataset(img_dir=test_images_dir, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Step 4: Read mapping.json to use correct image_id during predictions
    with open(mapping_path, 'r', encoding='utf-8') as f:
        filename_to_id = json.load(f)

    # Step 5: Perform inference and save prediction results to predictions.json
    predictions = []

    with torch.no_grad():
        for images, filenames in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for output, filename in zip(outputs, filenames):
                # Use mapping.json to get the corresponding image ID
                if filename not in filename_to_id:
                    print(f"Warning: {filename} not in mapping.json, skipping prediction.")
                    continue
                img_id = filename_to_id[filename]

                for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
                    box = box.cpu().numpy().tolist()
                    predictions.append({
                        'image_id': img_id,
                        'category_id': int(label.cpu().numpy()),
                        'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        'score': float(score.cpu().numpy())
                    })

    #  Write prediction results to a JSON file
    with open(predictions_output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Step 6: COCO evaluation
    coco_gt = COCO(generated_gt_path)
    coco_dt = coco_gt.loadRes(predictions_output_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    #Call summarize() but do not capture its return value
    coco_eval.summarize()

    # Directly retrieve metrics from coco_eval.stats
    stats = coco_eval.stats  # stats is usually a numpy array of length 12

    with open(os.path.join(output_dir, "scores.txt"), 'w', encoding='utf-8') as output_file:
       output_file.write("AP: {:.3f}\n".format(stats[0]))
       output_file.write("AP_0.5: {:.3f}\n".format(stats[1]))
       output_file.write("AP_0.75: {:.3f}\n".format(stats[2]))
       output_file.write("AP_small: {:.3f}\n".format(stats[3]))
       output_file.write("AP_medium: {:.3f}\n".format(stats[4]))
       output_file.write("AP_large: {:.3f}\n".format(stats[5]))


    print("Evaluation completed. Results saved to scores.txt")

if __name__ == "__main__":
    main()

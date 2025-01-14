# Machine-vision-apple-detection-and-counting
This repository contains the implementation of an apple detection and counting system, utilizing both traditional image processing techniques (Method A) and machine learning-based approaches (Method B). The project was developed to address challenges in apple detection under varying environmental conditions such as illumination, occlusion, and appearance diversity.

## Download Pre-trained Models and Results
You can download the pre-trained models and output results from the GitHub Releases section.

- [Download output.zip](https://github.com/JiajunGuo1027/Machine-vision-apple-detection-and-counting/releases/download/v1.0.0/output.zip)
- [Download saved_models.zip](https://github.com/JiajunGuo1027/Machine-vision-apple-detection-and-counting/releases/download/v2.0.0/saved_models.zip)

Please extract these files into the corresponding directories before running the scripts.

## **Features**
- **Approach A (Conventional Image Processing)**:
  - Enhanced image brightness and contrast using CLAHE.
  - HSV-based dynamic thresholding for apple segmentation.
  - Morphological operations to refine masks.
  - Contour analysis and watershed segmentation for occlusion handling.
  - Accurate detection and annotation of apples in images.

- **Approach (Machine Learning)**:
  - Apple detection using Faster R-CNN with ResNet-50 backbone.
  - Regression-based apple counting using ResNet-18.
  - Robust handling of overlapping apples and misdetections.
  - Evaluation metrics include MAE, MSE, \(R^2\) score, and Mean Accuracy.

## **Requirements**
The project requires the following dependencies:
- Python 3.12.8
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- pycocotools

## **Usage**
### **1. Run Approach A**
Navigate to the `method_a` folder and run the script:
```bash
python ImageProcessing_F.py
```
where `ImageProcessing_30.py` and `ImageProcessing_90.py` used different thresholds.

The output will include Apple detections with annotations.

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="Results/Image%20processing/FigureA1_90.png" alt="ImageProcessing_90 Result" style="width: 31%;">
  <img src="Results/Image%20processing/FigureA1_30.png" alt="ImageProcessing_30 Result" style="width: 31%;">
  <img src="Results/Image%20processing/FigureA1_79F.png" alt="ImageProcessing_F Result" style="width: 31%;">
</div>


### **2. Run Approach B**

#### **Training**
Navigate to the `method_b` folder and run the training scripts:
```bash
python train_rcnn_detection.py
python train_rcnn_counting.py
```

#### **Testing**
- Run the testing scripts to evaluate the detection models:
```bash
python test_all_formal.py
python test_MinneApple.py
```

- Run the testing scripts to evaluate the counting models:
```bash
python test_counting.py
```
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="Results/Some%20Machine%20learing%20Results/detected_dataset1_back_1.png" alt="ImageProcessing_90 Result" style="width: 31%;">
  <img src="Results/Some%20Machine%20learing%20Results/detected_dataset3_back_90.png" alt="ImageProcessing_30 Result" style="width: 31%;">
  <img src="Results/Some%20Machine%20learing%20Results/detected_dataset4_front_240.png" alt="ImageProcessing_F Result" style="width: 31%;">
</div>


#### **Quantitative Results**
| Metric           | Method B  |
|------------------|-----------|
| MAE             | 0.2237    |
| MSE             | 0.1884    |
| \(R^2\) Score   | 0.5762    |
| Mean Accuracy   | 0.8424    |

#### **Counting Metrics Visualization**
1. Scatter plot of predicted counts vs. ground truth counts

2. Histogram of prediction errors

3. Line plot of predicted vs. ground truth counts

4. Box plot of ground truth and predicted counts

<table>
  <tr>
    <td><img src="Results/Evaluation%20Machine%20learning/scatter%20diagram.png" alt="ImageProcessing_90 Result" style="width: 400px;"></td>
    <td><img src="Results/Evaluation%20Machine%20learning/error%20histogram.png" alt="ImageProcessing_30 Result" style="width: 400px;"></td>
  </tr>
  <tr>
    <td><img src="Results/Evaluation%20Machine%20learning/line%20plot.png" alt="ImageProcessing_F Result" style="width: 400px;"></td>
    <td><img src="Results/Evaluation%20Machine%20learning/box%20plots.png" alt="ImageProcessing_F Result" style="width: 400px;"></td>
  </tr>
</table>

## **Conclusion**
Both methods offer valuable insights into apple detection and counting:
- Approach A provides comprehensive detection but requires manual tuning and struggles with accuracy in complex scenarios.
- Approach B leverages machine learning for higher accuracy and is particularly suitable for automated harvesting tasks.
- Approach A has a wider detection range and may be suitable for applications that require a complete inventory of apples, including apples on the ground. However, because it relies on feature engineering, it is sensitive to environmental changes such as illumination and background complexity. In contrast, Approach B makes use of feature learning, which enables it to generalize under various conditions and is more suitable for practical application scenarios such as automatic picking. However, the model performance of Approach B depends on effective data preparation and model training, factors that directly affect its accuracy.

# Machine-vision-apple-detection-and-counting
This repository contains the implementation of an apple detection and counting system, utilizing both traditional image processing techniques (Method A) and machine learning-based approaches (Method B). The project was developed to address challenges in apple detection under varying environmental conditions such as illumination, occlusion, and appearance diversity.

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

![ImageProcessing_90 Result](Results/Image_processing/FigureA1_90.png)
![ImageProcessing_30 Result](Results/Image_processing/FigureA1_30.png)
![ImageProcessing_F Result](Results/Image_processing/FigureA1_79F.png)

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
#### **Quantitative Results**
| Metric           | Method B  |
|------------------|-----------|
| MAE             | 0.2237    |
| MSE             | 0.1884    |
| \(R^2\) Score   | 0.5762    |
| Mean Accuracy   | 0.8424    |

#### **Counting Metrics Visualization**
1. Scatter plot of predicted counts vs. ground truth counts:
   ![Scatter Plot](Results/Evaluation_Machine_learning/scatter_diagram.png)

2. Histogram of prediction errors:
   ![Histogram](Results/Evaluation_Machine_learning/error_histogram.png)

3. Line plot of predicted vs. ground truth counts:
   ![Line Plot](Results/Evaluation_Machine_learning/line_plot.png)

4. Box plot of ground truth and predicted counts:
   ![Box Plot](Results/Evaluation_Machine_learning/box_plots.png)


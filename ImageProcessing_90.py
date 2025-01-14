import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像 dataset1_back_1
image_path = 'C:/Users/31606/Desktop/MV_dataset/detection/test/images/dataset1_back_661.png'
image = cv2.imread(image_path)
original = image.copy()

# 图像预处理：增强亮度和对比度
def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

image = enhance_image(image)

# 转换到HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#动态阈值生成函数
def get_color_masks(hsv_image):
    # 红色区域
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) | cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # 近红色（橙色等）区域
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    
    # 综合红色和近红色
    combined_mask = cv2.bitwise_or(red_mask, orange_mask)
    return combined_mask

# def get_color_masks(hsv_image):
#     # 放宽红色和橙色的HSV范围
#     lower_red1 = np.array([0, 100, 80])
#     upper_red1 = np.array([10, 255, 255])
#     lower_red2 = np.array([170, 100, 80])
#     upper_red2 = np.array([180, 255, 255])

#     lower_orange = np.array([10, 80, 80])
#     upper_orange = np.array([25, 255, 255])
    
#     red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1) | cv2.inRange(hsv_image, lower_red2, upper_red2)
#     orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
#     combined_mask = cv2.bitwise_or(red_mask, orange_mask)
#     return combined_mask

# 获取颜色掩码
mask = get_color_masks(hsv)

# 自适应形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# 轮廓提取函数：加入多种特征融合
def process_contours(contours, mask, original_image):
    enhanced_mask = original_image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # 针对大目标进行细分
            x, y, w, h = cv2.boundingRect(contour)
            roi = mask[y:y + h, x:x + w]
            
            # 距离变换+分水岭分割
            dist_transform = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
            _, fg_mask = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
            fg_mask = np.uint8(fg_mask)
            
            # 未知区域
            unknown = cv2.subtract(roi, fg_mask)
            _, markers = cv2.connectedComponents(fg_mask)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(original_image[y:y + h, x:x + w], markers)
            
            # 标注分割结果
            for i in range(2, np.max(markers) + 1):
                mask_seg = np.uint8(markers == i)
                sub_contours, _ = cv2.findContours(mask_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for sub_c in sub_contours:
                    sub_area = cv2.contourArea(sub_c)
                    if sub_area > 50:  # 子区域的最小面积阈值
                        ((cx, cy), radius) = cv2.minEnclosingCircle(sub_c)
                        if radius > 5:
                            cv2.circle(enhanced_mask[y:y + h, x:x + w], (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        elif 50 < area < 300:  # 针对中等目标检测形状
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            if perimeter > 0:
                roundness = 4 * np.pi * (cv2.contourArea(hull) / (perimeter * perimeter))
                if roundness < 0.5:  # 遮挡区域标注
                    x, y, w, h = cv2.boundingRect(hull)
                    cv2.rectangle(enhanced_mask, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return enhanced_mask

# 提取轮廓并处理
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
processed_image = process_contours(contours, mask, original)

# 特征融合：纹理检测 (Laplacian 边缘) + 颜色特征
def texture_analysis(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_mask = cv2.inRange(laplacian, 20, 255)  # 设置边缘响应阈值
    combined = cv2.bitwise_and(mask, mask, mask=laplacian_mask)
    return combined

# 融合纹理和颜色掩码
fused_mask = texture_analysis(image, mask)

# 合并最终结果
final_result = cv2.bitwise_and(processed_image, processed_image, mask=fused_mask)

# 统计苹果数量并在原始图像上标注
def annotate_and_count_apples(contours, original_image):
    apple_count = 0
    annotated_image = original_image.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 排除小噪声区域
            apple_count += 1  # 每个有效轮廓视为一个苹果
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Apple {apple_count}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return annotated_image, apple_count

# 调用标注和统计函数
annotated_image, apple_count = annotate_and_count_apples(contours, original)

# 打印苹果数量并显示标注后的原始图像
print(f"Detected Apples: {apple_count}")

# 显示标注后的原始图像
plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title(f"Annotated Image with {apple_count} Apples")
plt.axis('off')
plt.show()

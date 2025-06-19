# preprocessor.py
import cv2
import numpy as np
import math


def denoise_before(img):
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    return denoised


def do_rotation(img, angle_range=45, padding=100, min_angle=0.5):
    # 如果图像太小，直接返回
    if img.shape[0] < 50 or img.shape[1] < 50:
        return img

    # Canny边缘检测
    edge = cv2.Canny(img, 50, 200, apertureSize=3)
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=80,
                        minLineLength=int(0.1*0.5*(img.shape[0]+img.shape[1])), maxLineGap=10)
    if lines is None:
        return img

    angles = []

    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:  # 垂直线忽略
            continue

        angle_deg = np.rad2deg(np.arctan2(dy, dx))
        if angle_deg > 90:
             angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180  # 转换到 [-90, 90]
        # 仅关注水平线，筛选
        if abs(angle_deg) > angle_range:
            continue

        angles.append(angle_deg)
        print(f"[DEBUG]angle = {angle_deg:.2f}!")

    if not angles:
        return img

    angles_rad = np.deg2rad(angles)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    dominant_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    print(f"[INFO] rotated: {dominant_angle:.2f}.")

    if abs (dominant_angle)<min_angle:
        return img


    img_padded = cv2.copyMakeBorder(
        img,
        top=padding,
        bottom=padding, left=padding,
        right=padding, borderType=cv2.BORDER_CONSTANT,
        value=255  # 白色边框
    )

    # 仿射旋转
    h, w = img_padded.shape[:2]
    center = (w // 2, h // 2)

    # 旋转矩阵
    M = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)

    # 计算新边界大小
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # 调整旋转中心
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 执行旋转
    rotated = cv2.warpAffine(img_padded, M, (new_w, new_h), borderValue=255)

    return rotated


def after_rotation(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return binary


def preprocess_image_from_array(img_array):
    """
    直接从内存中的图像数组进行预处理
    :param img_array: numpy数组形式的图像
    :return: 预处理后的图像
    """
    # 如果图像是彩色，转换为灰度
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_array

    if np.mean(img_gray) < 127:
        img_gray = cv2.bitwise_not(img_gray)  # 统一为白底黑字


    # 旋转前简单去噪
    denoised1 = denoise_before(img_gray)

    # 霍夫变换旋转矫正
    rotated = do_rotation(denoised1)

    # 旋转后处理
    final = after_rotation(rotated)

    return final
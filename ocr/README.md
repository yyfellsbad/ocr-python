# OCR 图像文字识别系统

基于 Python + OpenCV + Tesseract 的图像文字识别系统，支持图像预处理、文字识别与图形用户界面操作。

## 项目介绍

1. 支持图像文件拖拽或文件资源管理器选择

2. 图像预处理（灰度化、去噪、自动旋转矫正、二值化、文本块识别）

3. Tesseract OCR 引擎识别文本

4. 提供简洁图形界面

5. 可识别英文和简体中文字符

## 环境配置

0. python                    3.13.4
1. numpy                     2.2.3
2. opencv-python             4.11.0.86
3. pillow                    11.1.0
4. PyQt5                     5.15.11
5. pytesseract               0.3.13 (需要提前安装OCR引擎：https://github.com/UB-Mannheim/tesseract/wiki)

## 快速运行

直接运行`python main.py`即可

## 模块说明

- preprocessor.py:
    - denoise_before：提前做去噪处理
    - do_rotation：霍夫变换进行旋转识别和校正
    - after_rotation：旋转后处理
    - preprocess_image_from_array:主流程函数

- ocr.py:
    - blocks_detection：文本块识别，用于识别多列文本
    - run_ocr：识别文字

- main.py
    - 提供gui界面并运行上面的程序

## 示例效果

![alt text](readme_pic\\image.png)

先点击预处理，再进行ocr识别。可以选择保存结果，清空后可再次导入。

![alt text](readme_pic\\image-1.png)

## 联系方式

1558552900@qq.com
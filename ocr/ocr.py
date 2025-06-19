import pytesseract
import cv2

# 分中英文检测文本块
def blocks_detection(gray_img):
    # 高斯模糊平滑
    blur = cv2.GaussianBlur(gray_img, (7, 7), 0)
    
    # 自适应阈值 + 反色（二值化）
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 结构元素，竖直结构更适合列检测
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    #cv2.imshow("Dilate", dilate)
    #cv2.waitKey(0)
    # 查找轮廓
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 从左到右排序
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    blocks = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 200 and w > 20:  # 过滤小噪声
            blocks.append((x, y, w, h))
    
    print(f"[INFO] blocks number0: {len(blocks)}")
    return blocks

def blocks_detection_Chinese(gray_img):
    # 高斯模糊平滑
    blur = cv2.GaussianBlur(gray_img, (7, 7), 0)
    
    # 自适应阈值 + 反色（二值化）
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 结构元素，竖直结构更适合列检测
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(gray_img.shape[1]*0.02), 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    #cv2.imshow("Dilate", dilate)
    #cv2.waitKey(0)
    # 查找轮廓
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 从左到右排序
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])

    blocks = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 50 and w > 20:  # 过滤小噪声
            blocks.append((x, y, w, h))
    
    print(f"[INFO] blocks number1: {len(blocks)}")
    return blocks



def run_ocr (preprocessed_img, lang="chi_sim+eng"):

    results = []
    # 检测文本块
    blocks = blocks_detection(preprocessed_img)
    if len(blocks) == 0:
        blocks = blocks_detection_Chinese(preprocessed_img)
        for i, (x, y, w, h) in enumerate(blocks, 1):
            roi0 = preprocessed_img[y:y+h, x:x+w]
            # ocr识别    
            custom_config = r'--oem 3 --psm 3'  
            text = pytesseract.image_to_string(roi0, lang=lang, config=custom_config)
            results.append(text)
    else:
        for i, (x, y, w, h) in enumerate(blocks, 1):
            roi1 = preprocessed_img[y:y+h, x:x+w]
            # OCR识别
            custom_config = r'--oem 3 --psm 3'  
            text = pytesseract.image_to_string(roi1, lang=lang, config=custom_config)
            results.append(text)

    return results



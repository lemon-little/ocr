import cv2
import pytesseract
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 加载图像
def load_image(path):
    image = cv2.imread(path)
    return image

# 预处理图像（灰度化和二值化）
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# 寻找文字区域并标记，返回所有文字区域的列表
# 返回两种格式：一种为四个点的，一种为左上角坐标和宽高的
def find_text_regions(image):
    d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    text_regions_points = []
    text_regions_coords = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        rect_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        text_regions_points.append(rect_points)
        text_regions_coords.append((x, y, w, h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return text_regions_points, text_regions_coords

# 保存标记了文字区域的图像
def save_image(image, path):
    cv2.imwrite(path, image)

# 在图像中画出给定的矩形框，矩形由四个点表示
def draw_rectangle(image, points):
    pt1, pt2, pt3, pt4 = points
    cv2.line(image, pt1, pt2, (0, 0, 255), 2)
    cv2.line(image, pt2, pt3, (0, 0, 255), 2)
    cv2.line(image, pt3, pt4, (0, 0, 255), 2)
    cv2.line(image, pt4, pt1, (0, 0, 255), 2)
    return image

# 替换指定区域内的文字
def replace_text_in_region(image, coords, new_text):
    x, y, w, h = coords
    # 裁剪出指定区域
    cropped_img = image[y:y+h, x:x+w]
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 设置字体（需根据系统字体路径调整）
    # 动态设置字体大小，并选择支持中文的字体（需根据系统字体路径调整）
    font_size = max(10, int(h * 0.8))  # 根据区域高度动态设置字体大小
    font_size = h - 10
    try:
        # font = ImageFont.truetype("arial.ttf", h - 10)
        font = ImageFont.truetype("/root/autodl-fs/ocr/popular-fonts/微软雅黑.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # 清除原文字内容
    draw.rectangle([0, 0, w, h], fill=(255, 255, 255))

    # 绘制新的文字内容
    draw.text((5, 5), new_text, fill=(0, 0, 0), font=font)

    # 转换回OpenCV格式并替换到原图中
    updated_cropped_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image[y:y+h, x:x+w] = updated_cropped_img
    return image

if __name__ == "__main__":
    # 设置路径
    input_image_path = '/root/autodl-fs/ocr/000a3eb88193b0e076d87f86cb6b5fb6.jpg'
    output_image_path = '/root/autodl-fs/ocr/000a3eb88193b0e076d87f86cb6b5fb6_new.jpg'

    # 加载并处理图像
    image = load_image(input_image_path)
    binary_image = preprocess_image(image)

    # 加载GOT-OCR2_0模型
    tokenizer = AutoTokenizer.from_pretrained('stepfun-ai/GOT-OCR2_0', 
                                              trust_remote_code=True, 
                                              cache_dir='/root/autodl-fs/ocr',
                                              )
    model = AutoModel.from_pretrained('stepfun-ai/GOT-OCR2_0', 
                                      trust_remote_code=True,
                                      low_cpu_mem_usage=True, 
                                      device_map='cuda', 
                                      use_safetensors=True, 
                                      pad_token_id=tokenizer.eos_token_id,
                                      cache_dir='/root/autodl-fs/ocr',
                                      )
    model = model.eval().cuda()

    # 查找文字区域并在图像中标记
    text_regions_points, text_regions_coords = find_text_regions(binary_image)
    # print(f'文字区域列表（四个点表示）: {text_regions_points}')
    # print(f'文字区域列表（左上角坐标和宽高表示）: {text_regions_coords}')

    # # 使用GOT-OCR2_0模型识别指定区域的文字并替换
    # for i, coords in enumerate(text_regions_coords):
    #     x, y, w, h = coords
    #     ocr_box = f'[{x}, {y}, {x + w}, {y + h}]'
    #     res = model.chat(tokenizer, input_image_path, ocr_type='format', ocr_box=ocr_box)
    #     new_text = f'替换文本_{i}'  # 此处可根据需要生成类似风格的新文本
    #     print(f'区域 {i} 的识别结果: {res}，替换为: {new_text}')
    #     image = replace_text_in_region(image, coords, new_text)
    
    ocr_box = f'[{text_regions_coords[1][0]},{text_regions_coords[1][1]},{text_regions_coords[1][2]},{text_regions_coords[1][3]}]'
    print(ocr_box)
    res = model.chat(tokenizer, input_image_path, ocr_type='format', ocr_box=ocr_box)
    print(res)
    new_text = f'替换文本_123'
    image = replace_text_in_region(image, text_regions_coords[1], new_text)
    
    
    # 保存处理后的图像
    save_image(image, output_image_path)
    print(f'处理完成，已保存结果到{output_image_path}')

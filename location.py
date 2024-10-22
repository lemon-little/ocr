import cv2
import pytesseract

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

if __name__ == "__main__":
    # 设置路径
    input_image_path = '/root/autodl-fs/ocr/000a3eb88193b0e076d87f86cb6b5fb6.jpg'
    output_image_path = '/root/autodl-fs/ocr/000a3eb88193b0e076d87f86cb6b5fb6_1.jpg'

    # 加载并处理图像
    image = load_image(input_image_path)
    binary_image = preprocess_image(image)

    # 查找文字区域并在图像中标记
    text_regions_points, text_regions_coords = find_text_regions(binary_image)
    print(f'文字区域列表（四个点表示）: {text_regions_points}')
    print(f'文字区域列表（左上角坐标和宽高表示）: {text_regions_coords}')

    # 保存处理后的图像
    # save_image(binary_image, output_image_path)

    # print(f'处理完成，已保存结果到{output_image_path}')
    
     # 画出指定的矩形框（示例）
    if text_regions_points:
        image_with_red_box = draw_rectangle(image, text_regions_points[1])
        save_image(image_with_red_box, output_image_path)
    else:
        save_image(binary_image, output_image_path)

    print(f'处理完成，已保存结果到{output_image_path}')


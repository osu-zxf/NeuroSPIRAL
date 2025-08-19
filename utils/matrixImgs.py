from PIL import Image
import math
import os

# 设定图片的文件夹路径和目标图片大小
folder_path = "./result/imgs"
image_size = (150, 150)  # 假设每张图片的大小是100x100像素

# 获取所有图片文件名
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 计算行列数
num_images = len(image_files)
columns = int(math.sqrt(num_images))  # 按平方根近似分布列数
rows = math.ceil(num_images / columns)

# 创建一个新的空白图像用于放置拼接后的大图
combined_width = columns * image_size[0]
combined_height = rows * image_size[1]
combined_image = Image.new('RGB', (combined_width, combined_height))

# 将每张图片粘贴到大图的指定位置
for idx, file_path in enumerate(image_files):
    img = Image.open(file_path).resize(image_size)  # 调整图片大小
    x = (idx % columns) * image_size[0]
    y = (idx // columns) * image_size[1]
    combined_image.paste(img, (x, y))

# 保存或显示拼接后的大图
combined_image.save("combined_image.png")
combined_image.show()

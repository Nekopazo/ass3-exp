import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# ====== 你的数据路径 ======
root = "./data/raw/BeefCattle_Muzzle_Individualized"

# 获取所有 cattle 文件夹
classes = sorted([
    d for d in os.listdir(root)
    if os.path.isdir(os.path.join(root, d))
])

# 随机选 9 个 ID
selected_classes = random.sample(classes, 9)

images = []

for cls in selected_classes:
    cls_path = os.path.join(root, cls)

    # 选该ID下的一张图片
    img_name = random.choice(os.listdir(cls_path))
    img_path = os.path.join(cls_path, img_name)

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))   # 如果你论文用224就保持一致
    images.append(img)

# ====== 画 3x3 网格 ======
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(selected_classes[i], fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig("nine_muzzle_samples.png", dpi=300)
plt.show()

import torch
import torchvision

# from vgg import vgg
from AlexNet import AlexNet
from vgg19 import VGG19
import time
import os
from PIL import Image

# 要记得 替换自己剪枝后的cfg
# model = AlexNet(cfg=[41, 'M', 78, 'M', 119, 103, 4, 'M'])
model = VGG19(cfg=[20, 'Max', 30, 57, 'Max',
                   55, 109, 102, 99, 'Max',
                   97, 164, 158, 138, 'Max',
                   136, 129, 126, 77, 'Max'])
# 模型名要根据自己模型更改
model.load_state_dict(torch.load("vgg16_15_0.95_acc_max.pth"))

# 测试的路径与类别
path = "./Dog"
l = "狗"

imgs = os.listdir(path)
len_imgs = len(imgs)

# 总耗时
mean = 0
# 正确率
acc = 0

# 定义类别对应字典
# dist = {0: "飞机", 1: "汽车", 2: "鸟", 3: "猫", 4: "鹿", 5: "狗", 6: "青蛙", 7: "马", 8: "船", 9: "卡车"}
dist = {0: "猫", 1: "狗"}

for i in imgs:
    # 读取图像
    img = Image.open(path + "/" + i)
    # print(img)

    # 缩放、格式、归一化
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                ])
    image = transform(img)
    # 注意维度转换，单张图片
    image1 = torch.reshape(image, (1, 3, 224, 224))

    a = time.time()
    # 测试开关
    model.eval()
    # 节约性能
    with torch.no_grad():
        output = model(image1)
        # 转numpy格式,列表内取第一个
        # print(output)
        a1 = dist[output.argmax(1).numpy()[0]]

        if a1 == l:
            acc += 1
        print(a1, end=" ")
        mean += time.time() - a
        # img.show()

time_mean = mean / len_imgs
print("识别{}张图片，总耗时{}".format(len_imgs, mean))
print("平均耗时：{}".format(time_mean))
print("正确率：{}".format(acc / len_imgs))

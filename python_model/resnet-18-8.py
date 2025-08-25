import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

IMG_PATH = "../datasets/imagenet-2000/test-2000/ILSVRC2012_test_00000001.JPEG"
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()

# Preprocessing
pre = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
x = pre(Image.open(IMG_PATH).convert("RGB")).unsqueeze(0)

# Forward 到 layer2 输出
with torch.no_grad():
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)     # 此处输出为 [1, 128, 28, 28]
    feat_input = x.squeeze(0).cpu().numpy()

np.save("featuremaps/00000001.npy", feat_input.astype(np.float64))
print("Saved input to layer3.0:", feat_input.shape)

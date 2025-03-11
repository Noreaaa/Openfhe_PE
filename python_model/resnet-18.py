import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary  # 使用 torchinfo 打印模型结构

# 定义 Square 激活函数
def square_activation(x):
    return x ** 2

# 定义基础的 ResNet 块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入和输出的通道数不一致，需要使用 1x1 卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = square_activation(out)  # 使用 Square 激活函数
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 残差连接
        out = square_activation(out)  # 使用 Square 激活函数
        return out

# 定义 ResNet-18
class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  # 使用 Average Pooling

        # ResNet 层
        self.layer1 = self._make_layer(64, 2, stride=1)  # 2 个块
        self.layer2 = self._make_layer(128, 2, stride=2)  # 2 个块
        self.layer3 = self._make_layer(256, 2, stride=2)  # 2 个块
        self.layer4 = self._make_layer(512, 2, stride=2)  # 2 个块

        # 全局平均池化和全连接层
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个块可能下采样，其余块 stride=1
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = square_activation(x)  # 使用 Square 激活函数
        x = self.avgpool(x)  # 使用 Average Pooling

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 数据增强和预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 的均值和标准差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 加载 CIFAR-100 数据集
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# 创建 ResNet-18 模型
model = ResNet18(num_classes=100)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 打印模型结构
print("修改后的 ResNet-18 模型结构：")
summary(model, input_size=(3, 32, 32))  # 输入尺寸为 CIFAR-100 的 3x32x32

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 学习率衰减

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), targets.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # 每 100 个 batch 打印一次损失
            print(f'Epoch: {epoch}, Batch: {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), targets.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 训练和测试
best_accuracy = 0.0
for epoch in range(1, 121):  # 训练 120 个 epoch
    train(epoch)
    accuracy = test()
    scheduler.step()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # 保存最佳模型权重
        torch.save(model.state_dict(), 'resnet18_cifar100_best.pth')

# 保存模型结构
torch.save(model, 'resnet18_cifar100_architecture.pth')

print("训练完成！最佳测试准确率：{:.2f}%".format(best_accuracy))
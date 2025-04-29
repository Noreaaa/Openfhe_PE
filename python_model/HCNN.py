import os
import argparse
import datetime
import sys 
import time
import torchmetrics
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader 


from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
INPUT_C = 3
INPUT_H = 32
INPUT_W = 32
EPOCHS = 1

train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
max_test_acc = 0.0
sw = SummaryWriter(log_dir="./histories")

running_loss_history = []
running_acc_history = []
val_running_loss_history = []
val_running_acc_history = []

parser = argparse.ArgumentParser()
parser.add_argument('--act', required=True, choices=['square'])

args = parser.parse_args()
act_str = args.act


now = datetime.datetime.now()
md_str = now.strftime('%m%d')
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE_NAME = f"{os.path.splitext(os.path.basename(__file__))[0]}-{act_str}"
SAVE_MODEL_DIR_NAME = "saved_models"
BEST_MODEL_STATE_DICT_PATH = f"{CUR_DIR}/{SAVE_MODEL_DIR_NAME}/{BASE_FILE_NAME}-{md_str}-best.pt"


class Square(nn.Module):
    def forward(self,x):
        return torch.square(x)
    

if act_str == 'square':
    activation = Square()

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()

        conv1_list = []

        conv1_list.append(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1))

        conv1_list.append(nn.BatchNorm2d(32))
        conv1_list.append(activation)

        conv2_list = []


        conv2_list.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))

        conv2_list.append(nn.BatchNorm2d(64))
        conv2_list.append(activation)

        conv3_list = []

        conv3_list.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        conv3_list.append(nn.BatchNorm2d(128))
        conv3_list.append(activation)

        self.conv1 = nn.Sequential(*conv1_list)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Sequential(*conv2_list)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Sequential(*conv3_list)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)



        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)



        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
def test(model, data_loader):
    model.to(DEVICE)
    model.eval()  # Inference mode

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.max(outputs, dim=1)[1]
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = float(correct / total)
    print(f"Accuracy: {acc*100:.2f}")
    return acc


def eval(model, test_loader):
    model.to(DEVICE)
    model.eval()  # Inference mode

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            preds = torch.max(outputs, dim=1)[1]
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total


def train_one_epoch(epoch, model, train_loader, val_loader, loss_func, optimizer):
    model.train()

    total_step = len(train_loader)
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Clear the gradients of all parameters
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        batch_out = model(inputs)
        loss = loss_func(batch_out, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(batch_out, -1)
        running_loss += loss.cpu().item()
        running_corrects += torch.sum(preds == labels.data).cpu().item()
        train_acc(preds.cpu(), labels.cpu())

        # Print statistics (per 100 iterations and end of epoch)
        if (i + 1) % 100 == 0 or (i + 1) == total_step:
            print(f"Step [{i+1:3d}/{total_step:3d}] -> Loss: {loss.item():.4f}")
    else:
        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(DEVICE)
                val_labels = val_labels.to(DEVICE)
                val_out = model(val_inputs)
                val_loss = loss_func(val_out, val_labels)

                val_preds = torch.argmax(val_out, dim=-1)
                val_running_loss += val_loss.cpu().item()
                val_running_corrects += torch.sum(val_preds == val_labels.data).cpu().item()
                test_acc(val_preds.cpu(), val_labels.cpu())

        epoch_loss = running_loss / len(train_loader)  # loss per epoch
        epoch_acc = running_corrects / len(train_loader.dataset)  # accuracy per epoch
        running_loss_history.append(epoch_loss)
        running_acc_history.append(epoch_acc)

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_acc = val_running_corrects / len(val_loader.dataset)
        val_running_loss_history.append(val_epoch_loss)
        val_running_acc_history.append(val_epoch_acc)

        print(f'training loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')
        print(f'validation loss: {val_epoch_loss:.4f}, validation acc: {val_epoch_acc:.4f}')
        print(f'TrainAcc: {train_acc.compute()}, TestAcc: {test_acc.compute()}')
        global max_test_acc
        if max_test_acc < test_acc.compute():
            print(f'Test acc improved from {max_test_acc} to {test_acc.compute()}')

            torch.save(model.state_dict(), BEST_MODEL_STATE_DICT_PATH)
            print('Model saved.')
            max_test_acc = test_acc.compute()
        sw.add_scalar('Train Accuracy', train_acc.compute(), epoch+1)
        sw.add_scalar('Test Accuracy', test_acc.compute(), epoch+1)



def main():
    global max_test_acc
    print(f"Device: {DEVICE}\n")

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=(0.5,), std=(0.5,))])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    #                               transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])



    train_data = datasets.CIFAR10(root="../cifar-10/data", train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root="../cifar-10/data", train=False, download=True, transform=transform)
    
    print("<Train data>")
    print(train_data)
    print()
    print(f"Train data images: {train_data.data.shape}")
    print(f"Train data labels: {len(train_data.targets)}\n")
    print("<Test data>")
    print(test_data)
    print()
    print(f"Test data images: {test_data.data.shape}")
    print(f"Test data labels: {len(test_data.targets)}\n")

    loaders = {
        "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    model = CifarCNN()

    summary(model, input_size=(1, INPUT_C, INPUT_H, INPUT_W))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    print()

    model = model.to(DEVICE)

    for epoch in range(EPOCHS):
        print(f"Epoch [{epoch+1:3d}/{EPOCHS:3d}]")
        train_acc.reset()
        test_acc.reset()
        train_one_epoch(epoch, model, loaders["train"], loaders["test"], loss_func, optimizer)

    best_model = CifarCNN()
    best_model.load_state_dict(torch.load(BEST_MODEL_STATE_DICT_PATH, map_location=torch.device(DEVICE)))
    best_model = best_model.to(DEVICE)
    start = time.perf_counter()
    test_accuracy = test(best_model, loaders["test"])
    end = time.perf_counter()
    print(f"Test accuracy: {test_accuracy}")
    
    for name, param in model.named_parameters():
        np.save(f"./saved_models/parameters/{name}.npy", param.cpu().detach().numpy())
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            np.save(f"{name}_running_mean.npy", module.running_mean.cpu().numpy())
            np.save(f"{name}_running_var.npy", module.running_var.cpu().numpy())


if __name__ == '__main__':
    main()

        

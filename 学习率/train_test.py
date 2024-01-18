import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用标号为1的GPU

import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from model import mobile_vit_xx_small as create_model
# from model import mobile_vit_x_small as create_model
# from model import mobile_vit_small as create_model
from utils import *

# 制表
import pandas as pd

# 画图
import matplotlib.pyplot as plt

Train_Loss = []
Train_Acc = []
Test_Loss = []
Test_Acc = []


def train(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} to train'.format(device))

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    batch_size = opt.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 下载数据集
    train_datasets = torchvision.datasets.CIFAR10('data', train=True, transform=transform, download=True)
    test_datasets = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=True)

    # 加载数据集
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=nw,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=nw,
                                pin_memory=True)

    model = create_model(num_classes=opt.num_classes).to(device)

    # 加载权重
    if opt.weights != "":
        assert os.path.exists(opt.weights), "weights file: '{}' not exist.".format(opt.weights)
        weights_dict = torch.load(opt.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 是否冻结权重
    if opt.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 定义优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=opt.lr, weight_decay=1E-2)

    best_acc = 0.

    # 开始训练和验证
    for epoch in range(opt.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=device,
                                                epoch=epoch)

        # test
        test_loss, test_acc = test(model=model,
                                     data_loader=test_dataloader,
                                     device=device,
                                     epoch=epoch)

        Train_Loss.append(round(train_loss, 4))
        Train_Acc.append(round(train_acc, 4))
        Test_Loss.append(round(test_loss, 4))
        Test_Acc.append(round(test_acc, 4))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")

        torch.save(model.state_dict(), "./weights/latest_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--lr', type=float, default=0.002)
    # parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--data-path', type=str, default="./data/cifar-10-batches-py")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./mobilevit_xxs.pt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    print(opt)
    train(opt)

    # 制表
    col1 = "Train_Loss"
    col2 = "Train_Acc"
    col3 = "Test_Loss"
    col4 = "Test_Acc"
    data = pd.DataFrame({col1: Train_Loss, col2: Train_Acc, col3: Test_Loss, col4: Test_Acc})
    data.to_excel("0.0001.xlsx", index=True)
    # data.to_excel("0.002.xlsx", index=True)
    # data.to_excel("0.02.xlsx", index=True)

    # 制图
    x = range(opt.epochs)

    # 训练时
    # 损失率
    plt.figure(num=1, figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x, Train_Loss, linewidth=2, color='r', marker='o',
             markerfacecolor='blue', markersize=7)
    plt.title('Train_Loss')
    for a, b in zip(x, Train_Loss):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # 准确率
    plt.subplot(2, 1, 2)
    plt.plot(x, Train_Acc, linewidth=2, color='r', marker='o',
             markerfacecolor='blue', markersize=7)
    plt.title('Train_Acc')
    for a, b in zip(x, Train_Acc):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("./0.0001_train.jpg")
    # plt.savefig("./0.002_train.jpg")
    # plt.savefig("./0.02_train.jpg")

    # 验证时
    # 损失率
    plt.figure(num=2, figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x, Test_Loss, linewidth=2, color='r', marker='o',
             markerfacecolor='blue', markersize=7)
    plt.title('Test_Loss')
    for a, b in zip(x, Test_Loss):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    # 准确率
    plt.subplot(2, 1, 2)
    plt.plot(x, Test_Acc, linewidth=2, color='r', marker='o',
             markerfacecolor='blue', markersize=7)
    plt.title('Test_Acc')
    for a, b in zip(x, Test_Acc):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("./0.0001_test.jpg")
    # plt.savefig("./0.002_test.jpg")
    # plt.savefig("./0.02_test.jpg")

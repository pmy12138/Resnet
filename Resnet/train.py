import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # CIFAR-10数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # 新增
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 新增
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2023, 0.1994, 0.2010])
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    }

    # 数据路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "."))
    image_path = os.path.join(data_root, "data")

    # 加载CIFAR-10
    import torchvision
    train_dataset = torchvision.datasets.CIFAR10(
        root=image_path,
        train=True,
        download=False,
        transform=data_transform["train"]
    )

    # 9:1划分
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    train_num = train_size
    val_num = val_size

    # CIFAR-10类别映射
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    cla_dict = {idx: class_name for idx, class_name in enumerate(cifar10_classes)}
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 10类模型,不加载预训练权重
    net = resnet34(num_classes=10)
    net.to(device)

    # 标签平滑损失
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Adam优化器配置
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0003, weight_decay=0.01, amsgrad=True)  #

    epochs = 80
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)

    writer = SummaryWriter('logs/cifar10_resnet34')
    accumulation_steps = 4  # 梯度累积步数
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_correct = 0  # 新增:训练准确样本数
        train_total = 0  # 新增:训练总样本数
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss = loss / accumulation_steps  # 归一化损失
            loss.backward()
            optimizer.step()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics
            running_loss += loss.item()
            # 计算训练准确率
            predict_y = torch.max(logits, dim=1)[1]
            train_correct += torch.eq(predict_y, labels.to(device)).sum().item()
            train_total += labels.size(0)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            epoch_train_loss = running_loss / train_steps
            epoch_train_acc = train_correct / train_total
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
            #     epoch + 1, epochs, loss * accumulation_steps
            # )

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_correct = 0
        val_total = 0
        val_loss = 0.0  # 新增:验证损失
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()

                predict_y = torch.max(outputs, dim=1)[1]
                val_correct += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_total += val_labels.size(0)

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        epoch_val_loss = val_loss / len(validate_loader)
        epoch_val_acc = val_correct / val_total

        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss: %.3f val_acc: %.3f' %
              (epoch + 1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))


        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(net.state_dict(), save_path)

    writer.close()
    print('Finished Training')


if __name__ == '__main__':
    main()

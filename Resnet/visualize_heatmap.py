import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model_GN_WS import resnet18_gn_ws  # 或您使用的模型

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class FeatureExtractor:
    """特征提取器:用于获取中间层的特征图"""

    def __init__(self, model, target_layers):
        """
        Args:
            model: ResNet模型
            target_layers: 要提取特征的层名称列表,如 ['layer1', 'layer4']
        """
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []

        # 注册前向钩子
        for name, module in model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(self._get_features(name))
                self.hooks.append(hook)

    def _get_features(self, name):
        """创建钩子函数,保存指定层的输出"""

        def hook(module, input, output):
            self.features[name] = output.detach()

        return hook

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()


def generate_heatmap(feature_map):
    """
    将特征图转换为热力图

    Args:
        feature_map: 形状为 (C, H, W) 的特征图

    Returns:
        热力图,形状为 (H, W),值范围 [0, 1]
    """
    # 对所有通道求平均,得到 (H, W)
    heatmap = torch.mean(feature_map, dim=0).cpu().numpy()

    # 归一化到 [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    将热力图叠加到原始图像上

    Args:
        image: 原始图像,形状 (H, W, 3),值范围 [0, 1]
        heatmap: 热力图,形状 (H, W),值范围 [0, 1]
        alpha: 热力图透明度

    Returns:
        叠加后的图像
    """
    # 将热力图调整到图像大小
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # 应用颜色映射(jet colormap)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # 叠加
    overlayed = alpha * heatmap_colored + (1 - alpha) * image

    return overlayed


def visualize_model_attention(model_path, data_root='./data', num_samples=5):
    """
    可视化模型的浅层和深层注意力区域

    Args:
        model_path: 训练好的模型权重路径
        data_root: CIFAR-10 数据集路径
        num_samples: 可视化的样本数量
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    net = resnet18_gn_ws(num_classes=10)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.to(device)
    net.eval()

    # 数据预处理(不包含归一化,便于可视化)
    transform_vis = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载测试集
    test_dataset = datasets.CIFAR10(root=data_root, train=False,
                                    download=False, transform=transform_vis)

    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 定义要提取的层:浅层(layer1)和深层(layer4)
    target_layers = ['layer1', 'layer4']
    extractor = FeatureExtractor(net, target_layers)

    # 可视化多个样本
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for idx in range(num_samples):
        # 获取样本
        image, label = test_dataset[idx]
        image_np = image.permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)

        # 前向传播(需要归一化)
        image_normalized = transforms.Normalize(
            [0.4914, 0.4822, 0.4465],
            [0.2023, 0.1994, 0.2010]
        )(image)

        with torch.no_grad():
            output = net(image_normalized.unsqueeze(0).to(device))
            pred_class = torch.argmax(output, dim=1).item()

            # 获取特征图
        shallow_features = extractor.features['layer1'][0]  # (C, H, W)
        deep_features = extractor.features['layer4'][0]  # (C, H, W)

        # 生成热力图
        shallow_heatmap = generate_heatmap(shallow_features)
        deep_heatmap = generate_heatmap(deep_features)

        # 叠加热力图
        shallow_overlay = overlay_heatmap(image_np, shallow_heatmap)
        deep_overlay = overlay_heatmap(image_np, deep_heatmap)

        # 绘制
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f'Original\nTrue: {class_names[label]}\nPred: {class_names[pred_class]}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(shallow_heatmap, cmap='jet')
        axes[idx, 1].set_title('Layer1 (Shallow)\nHeatmap')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(shallow_overlay)
        axes[idx, 2].set_title('Layer1 Overlay')
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(deep_overlay)
        axes[idx, 3].set_title('Layer4 (Deep) Overlay')
        axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig(model_path+' - heatmaps.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 清理钩子
    extractor.remove_hooks()

    print("热力图已保存到{} - heatmaps.png".format(model_path))


def compare_models_attention(model_paths, model_names, data_root='./data'):
    """
    对比多个模型的注意力区域

    Args:
        model_paths: 模型权重路径列表
        model_names: 模型名称列表
        data_root: 数据集路径
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载测试样本
    transform_vis = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root=data_root, train=False,
                                    download=False, transform=transform_vis)
    image, label = test_dataset[0]
    image_np = image.permute(1, 2, 0).numpy()

    # 对比可视化
    fig, axes = plt.subplots(len(model_paths), 3, figsize=(12, 4 * len(model_paths)))

    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        # 加载模型
        net = resnet18_gn_ws(num_classes=10)
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        net.to(device)
        net.eval()

        # 提取特征
        extractor = FeatureExtractor(net, ['layer1', 'layer4'])

        image_normalized = transforms.Normalize(
            [0.4914, 0.4822, 0.4465],
            [0.2023, 0.1994, 0.2010]
        )(image)

        with torch.no_grad():
            net(image_normalized.unsqueeze(0).to(device))

        shallow_heatmap = generate_heatmap(extractor.features['layer1'][0])
        deep_heatmap = generate_heatmap(extractor.features['layer4'][0])

        # 绘制
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'{model_name}\nOriginal')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(overlay_heatmap(image_np, shallow_heatmap))
        axes[i, 1].set_title('Layer1 (Shallow)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlay_heatmap(image_np, deep_heatmap))
        axes[i, 2].set_title('Layer4 (Deep)')
        axes[i, 2].axis('off')

        extractor.remove_hooks()

    plt.tight_layout()
    plt.savefig('models_comparison_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 单模型可视化
    visualize_model_attention(
        model_path='./resnet18_GN_WS.2.pth',
        data_root='./data',
        num_samples=5
    )

    # # 多模型对比(如果有多个模型)
    # compare_models_attention(
    #     model_paths=['./resnet34.pth', './resnet18_GN_WS.pth'],
    #     model_names=['ResNet34', 'ResNet18-GN-WS'],
    #     data_root='./data'
    # )
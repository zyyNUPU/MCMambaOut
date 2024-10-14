import os
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from train_classification import MRIRandomCrop2D,MRIRandomHorizontalFlip2D,MRIToTensor2D,MRIResize2D,MRIDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import  AttributesDataset, mean, std
from model import MultiOutputModel_test

def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))

def get_model_outputs_and_labels(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    all_labels = {}
    all_preds = {}

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['img'].to(device)
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}

            # 获取模型输出
            outputs = model(imgs)
            
            # 存储预测概率和真实标签
            for key in outputs:
                label_key = f'{key}_labels'
                
                outputs[key] = torch.nn.functional.softmax(outputs[key], dim=1)

                if len(all_preds) == 0:
                    all_preds[key] = [outputs[key].cpu().numpy()]
                    all_labels[key] = [target_labels[label_key].cpu().numpy()]
                else:
                    if key in all_preds:
                        all_preds[key].append(outputs[key].cpu().numpy())
                        all_labels[key].append(target_labels[label_key].cpu().numpy())
                    else:
                        all_preds[key] = [outputs[key].cpu().numpy()]
                        all_labels[key] = [target_labels[label_key].cpu().numpy()]

    # 合并所有批次的预测和标签
    all_preds = {key: np.concatenate(val) for key, val in all_preds.items()}
    all_labels = {key: np.concatenate(val) for key, val in all_labels.items()}

    print(all_preds['noise'].shape)
    print(all_labels['noise'].shape)
    

    return all_labels, all_preds

plt.rcParams.update({'font.size': 18})

def plot_roc_curve(y_true, y_score, labels, output_path):
    n_classes = y_score[list(y_score.keys())[0]].shape[1]  # 获取类别数
    print(n_classes)
    for i, label in enumerate(labels):
        y_true_bin = label_binarize(y_true[label], classes=[0, 1, 2])
        y_score_bin = y_score[label]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for j in range(n_classes):
            fpr[j], tpr[j], _ = roc_curve(y_true_bin[:, j], y_score_bin[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        plt.figure(figsize=(10, 8))
        colors = ['aqua', 'darkorange', 'cornflowerblue']
        for j, color in zip(range(n_classes), colors):
            plt.plot(fpr[j], tpr[j], color=color, lw=2, label=f'ROC curve for class {j} (area = {roc_auc[j]:0.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate',fontsize=24)
        plt.ylabel('True Positive Rate',fontsize=24)
        plt.title(f'Receiver Operating Characteristic for {label}',fontsize=24)
        plt.legend(loc="lower right")

        plt.savefig(f'{output_path}/{label}_roc_curve.png', dpi=300)
        plt.close()

# 示例调用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = {
    "train": transforms.Compose([
    # 自定义的裁剪方法，可以根据需要进行调整
    MRIRandomCrop2D(size=(92, 106)),
    MRIRandomHorizontalFlip2D(),
    MRIToTensor2D(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # 根据实际数据情况进行调整
]),
    "val": transforms.Compose([
    MRIResize2D(size=(92, 106)),
    MRIToTensor2D(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # 根据实际数据情况进行调整
])
}
val_file_name = "val_patch"
image_path = os.path.join("/data/zyy", "MICCAI_challenge")
output_path = '/home/zyy/Modules/Classification/PyTorch-Multi-Label-Image-Classification/figure/MambaOut_09'
checkpoint = "/home/zyy/Modules/Classification/PyTorch-Multi-Label-Image-Classification/parameter/MambaOut_muti_pre09.pth"
labels = ['noise', 'zipper', 'positioning', 'banding', 'motion', 'contrast', 'distortion']
batch_size = 36
num_workers = 8 
attributes = AttributesDataset('/data/zyy/MICCAI_challenge/LISA_LF_QC_updated.csv')
val_dataset = MRIDataset(root_dir=os.path.join(image_path, val_file_name),attributes=attributes,transform=data_transform["val"])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
model = MultiOutputModel_test(n_classes=3,device=device).to(device)
checkpoint_load(model, checkpoint)
# 获取模型输出和真实标签
y_true, y_score = get_model_outputs_and_labels(model, val_dataloader, device)

# 绘制ROC曲线
plot_roc_curve(y_true, y_score, labels, output_path)

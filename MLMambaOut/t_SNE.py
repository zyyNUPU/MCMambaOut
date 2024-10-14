import os
from sklearn.decomposition import PCA
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from train_classification import MRIRandomCrop2D,MRIRandomHorizontalFlip2D,MRIToTensor2D,MRIResize2D,MRIDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import  AttributesDataset, mean, std
from model import MultiOutputModel_test
import matplotlib.colors as mcolors

plt.rcParams.update({'font.size': 22})

def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))

def extract_features_for_labels(model, dataloader, device):
    model.eval()  # 设置模型为评估模式
    all_features = {label: [] for label in ['noise', 'zipper', 'positioning', 'banding', 'motion', 'contrast', 'distortion']}
    all_labels = {label: [] for label in ['noise', 'zipper', 'positioning', 'banding', 'motion', 'contrast', 'distortion']}

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['img'].to(device)
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            
            # 获取模型特征
            # features = model.base_model(imgs)  # 提取特征
            # features = model.pool(features)     # 进行池化操作
            # features = torch.flatten(features, 1)  # 拉平以适应后续处理
            # for key in all_features:
            #     if len(all_features[key]) == 0:
            #         all_features[key].append(features.cpu().numpy())
            #         all_labels[key].append(target_labels[f'{key}_labels'].cpu().numpy())
            #     else:
            #         all_features[key].append(features.cpu().numpy())
            #         all_labels[key].append(target_labels[f'{key}_labels'].cpu().numpy())
            
            
            #features = model(imgs)
            #print(features.keys())  # 打印所有键，以了解字典中包含哪些张量

            
            features = model(imgs)
            for key in all_features:
                if len(all_features[key]) == 0:
                    all_features[key].append(features[key].cpu().numpy())
                    all_labels[key].append(target_labels[f'{key}_labels'].cpu().numpy())
                else:
                    all_features[key].append(features[key].cpu().numpy())
                    all_labels[key].append(target_labels[f'{key}_labels'].cpu().numpy())
            

    # 合并所有特征和标签
    for key in all_features:
        all_features[key] = np.concatenate(all_features[key], axis=0)
        all_labels[key] = np.concatenate(all_labels[key], axis=0)

    print(all_features['noise'].shape)
    print(all_labels['noise'].shape)
    
    return all_features, all_labels


def plot_tsne_for_labels(features_dict, labels_dict, output_path_prefix):
    for label in features_dict:
        features = features_dict[label]
        labels = labels_dict[label]

        

        tsne = TSNE(n_components=2, perplexity=30, n_iter=300,random_state=0,learning_rate=100)
        reduced_features = tsne.fit_transform(features)
        
        

        plt.figure(figsize=(12, 10))
        colors = ['r', 'g', 'b']  # 例如红色、绿色、蓝色分别代表0,1,2
        cmap = mcolors.ListedColormap(colors)
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=cmap, alpha=0.7)
        
        #scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='Set1', alpha=0.7)
        plt.colorbar(scatter, ticks=np.arange(3), label='Classes')
        plt.xlabel('t-SNE Component 1',fontsize=30)
        plt.ylabel('t-SNE Component 2',fontsize=30)
        plt.title(f't-SNE - {label}',fontsize=34)
        plt.savefig(f'{output_path_prefix}_{label}_tsne_plot.png', dpi=300)
        plt.close()

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
val_file_name = "train_patch"
image_path = os.path.join("/data/zyy", "MICCAI_challenge")
output_path = '/home/zyy/Modules/Classification/PyTorch-Multi-Label-Image-Classification/figure/MambaOut_04_tSNE/MambaOut'
checkpoint = "/home/zyy/Modules/Classification/PyTorch-Multi-Label-Image-Classification/parameter/MambaOut_muti_pre04.pth"
labels = ['noise', 'zipper', 'positioning', 'banding', 'motion', 'contrast', 'distortion']
batch_size = 36
num_workers = 8 
attributes = AttributesDataset('/data/zyy/MICCAI_challenge/LISA_LF_QC_updated.csv')
val_dataset = MRIDataset(root_dir=os.path.join(image_path, val_file_name),attributes=attributes,transform=data_transform["val"])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
model = MultiOutputModel_test(n_classes=3,device=device).to(device)
checkpoint_load(model, checkpoint)


# 获取每个标签的特征和标签
features_dict, labels_dict = extract_features_for_labels(model, val_dataloader, device)

# 绘制每个标签的t-SNE图
plot_tsne_for_labels(features_dict, labels_dict, output_path)

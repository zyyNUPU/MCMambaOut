import csv
import os
import nibabel as nib
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class AttributesDataset():
    def __init__(self, annotation_path):
        noise_labels = []
        zipper_labels = []
        positioning_labels = []
        banding_labels = []
        motion_labels = []
        contrast_labels = []
        distortion_labels = []
        

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                noise_labels.append(row['Noise'])
                zipper_labels.append(row['Zipper'])
                positioning_labels.append(row['Positioning'])
                banding_labels.append(row['Banding'])
                motion_labels.append(row['Motion'])
                contrast_labels.append(row['Contrast'])
                distortion_labels.append(row['Distortion'])

        self.Noise_labels = np.unique(noise_labels)
        self.Zipper_labels = np.unique(zipper_labels)
        self.Positioning_labels = np.unique(positioning_labels)
        self.Banding_labels = np.unique(banding_labels)
        self.Motion_labels = np.unique(motion_labels)
        self.Contrast_labels = np.unique(contrast_labels)
        self.Distortion_labels = np.unique(distortion_labels)

        self.num_noise = len(self.Noise_labels)
        self.num_zipper = len(self.Zipper_labels)
        self.num_positioning = len(self.Positioning_labels)
        self.num_banding = len(self.Banding_labels)
        self.num_motion = len(self.Motion_labels)
        self.num_contrast = len(self.Contrast_labels)
        self.num_distortion = len(self.Distortion_labels)


        self.Noise_id_to_name = dict(zip(range(len(self.Noise_labels)), self.Noise_labels))
        self.Noise_name_to_id = dict(zip(self.Noise_labels, range(len(self.Noise_labels))))

        self.Zipper_id_to_name = dict(zip(range(len(self.Zipper_labels)), self.Zipper_labels))
        self.Zipper_name_to_id = dict(zip(self.Zipper_labels, range(len(self.Zipper_labels))))

        self.Positioning_id_to_name = dict(zip(range(len(self.Positioning_labels)), self.Positioning_labels))
        self.Positioning_name_to_id = dict(zip(self.Positioning_labels, range(len(self.Positioning_labels))))

        self.Banding_id_to_name = dict(zip(range(len(self.Banding_labels)), self.Banding_labels))
        self.Banding_name_to_id = dict(zip(self.Banding_labels, range(len(self.Banding_labels))))

        self.Motion_id_to_name = dict(zip(range(len(self.Motion_labels)), self.Motion_labels))
        self.Motion_name_to_id = dict(zip(self.Motion_labels, range(len(self.Motion_labels))))

        self.Contrast_id_to_name = dict(zip(range(len(self.Contrast_labels)), self.Contrast_labels))
        self.Contrast_name_to_id = dict(zip(self.Contrast_labels, range(len(self.Contrast_labels))))

        self.Distortion_id_to_name = dict(zip(range(len(self.Distortion_labels)), self.Distortion_labels))
        self.Distortion_name_to_id = dict(zip(self.Distortion_labels, range(len(self.Distortion_labels))))


class MRIDataset(Dataset):
    def __init__(self, root_dir, attributes,transform=None):
        self.root_dir = root_dir
        self.attr = attributes
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.file_paths = []
        self.data = []  # 用于预加载数据
        self.noise_labels = []
        self.zipper_labels = []
        self.positioning_labels = []
        self.banding_labels = []
        self.motion_labels = []
        self.contrast_labels = []
        self.distortion_labels = []
        csv_file = "/data/zyy/MICCAI_challenge/LISA_LF_QC_updated.csv"
        df = pd.read_csv(csv_file)

        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.endswith('.nii.gz'):
                    file_path = os.path.join(cls_path, fname)
                    prefix_name = '_'.join(fname.split('_')[:4])
                    table_file_name = prefix_name + ".nii.gz"
                    row = df[df['filename'] == table_file_name]
                    self.noise_labels.append(self.attr.Noise_name_to_id[str(row['Noise'].values[0])])
                    self.zipper_labels.append(self.attr.Zipper_name_to_id[str(row['Zipper'].values[0])])
                    self.positioning_labels.append(self.attr.Positioning_name_to_id[str(row['Positioning'].values[0])])
                    self.banding_labels.append(self.attr.Banding_name_to_id[str(row['Banding'].values[0])])
                    self.motion_labels.append(self.attr.Motion_name_to_id[str(row['Motion'].values[0])])
                    self.contrast_labels.append(self.attr.Contrast_name_to_id[str(row['Contrast'].values[0])])
                    self.distortion_labels.append(self.attr.Distortion_name_to_id[str(row['Distortion'].values[0])])
                    #label = self.class_to_idx[cls_name]
                    img = nib.load(file_path).get_fdata()
                    self.data.append(img)
                    self.file_paths.append(file_path)

        self.total_slices = len(self.data)

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'noise_labels': self.noise_labels[idx],
                'zipper_labels': self.zipper_labels[idx],
                'positioning_labels': self.positioning_labels[idx],
                'banding_labels': self.banding_labels[idx],
                'motion_labels': self.motion_labels[idx],
                'contrast_labels': self.contrast_labels[idx],
                'distortion_labels': self.distortion_labels[idx]
            }
        }
        return dict_data

class MRIDataset_val(Dataset):
    def __init__(self, root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.file_paths = []
        self.data = []  
        self.file_name = []
        
        for fname in os.listdir(root_dir):
            if fname.endswith('.nii.gz'):
                file_path = os.path.join(root_dir, fname)
                img = nib.load(file_path).get_fdata()
                self.data.append(img)
                self.file_paths.append(file_path)
                self.file_name.append(fname)

        self.total_slices = len(self.data)

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        filename = self.file_name[idx]
        dict_data = {
            'img': img,
            'filename': filename
        }
        return dict_data




# ad=AttributesDataset("/data/zyy/MICCAI_challenge/LISA_LF_QC_updated.csv")
# print(ad.Noise_id_to_name)
# print(ad.Noise_name_to_id)
# print(ad.Zipper_id_to_name)
# print(ad.Positioning_id_to_name)
# print(ad.Banding_id_to_name)
# print(ad.Motion_id_to_name)
# print(ad.Contrast_id_to_name)
# print(ad.Distortion_id_to_name)


# train_file_name = "train_patch"
# val_file_name = "val_patch"

# image_path = os.path.join("/data/zyy", "MICCAI_challenge")  # flower data set path
# assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

# train_dataset = MRIDataset(root_dir=os.path.join(image_path, train_file_name),attributes=ad,transform=None)

# print(train_dataset.__getitem__(idx=4444)['labels'])
# print(train_dataset.file_paths[4444])

# print(train_dataset.__getitem__(idx=5555)['labels'])
# print(train_dataset.file_paths[5555])

# print(train_dataset.__getitem__(idx=6666)['labels'])
# print(train_dataset.file_paths[6666])

# print(train_dataset.__getitem__(idx=7777)['labels'])
# print(train_dataset.file_paths[7777])
import argparse
import os
from datetime import datetime
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import  AttributesDataset, mean, std
from model import MultiOutputModel,MultiOutputModel_decoder_new_loss
from test import calculate_metrics, validate, visualize_grid
from torch.utils.data import DataLoader
from dataset import MRIDataset

class MRIRandomCrop2D(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):

        height, width = image.shape
        

        h_start = np.random.randint(0, height - self.size[0])
        w_start = np.random.randint(0, width - self.size[1])
        

        image = image[h_start:h_start+self.size[0],
                      w_start:w_start+self.size[1]]
        
        return image

class MRIResize2D(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):

        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        resized_image = torch.nn.functional.interpolate(image_tensor, size=self.size, mode='nearest').squeeze(0).squeeze(0)
        
        return resized_image.numpy()


class MRIRandomHorizontalFlip2D(object):
    def __call__(self, image):
        
        

        if np.random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
        
        return image

class MRIToTensor2D(object):
    def __call__(self, image):
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        
        return image

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    torch.save(model.state_dict(), name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='./LISA_LF_QC_updated.csv',
                        help="Path to the file with attributes")
    args = parser.parse_args()

    start_epoch = 1
    N_epochs = 500
    batch_size = 36
    num_workers = 8  # number of processes to handle dataset loading
    lr=0.00015

    train_file_name = "train_patch" 
    val_file_name = "val_patch"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)
    

    # specify image transforms for augmentation during training
    data_transform = {
        "train": transforms.Compose([

        MRIRandomCrop2D(size=(92, 106)),
        MRIRandomHorizontalFlip2D(),
        MRIToTensor2D(),
        transforms.Normalize(mean=[0.485], std=[0.229])  
    ]),
        "val": transforms.Compose([
        MRIResize2D(size=(92, 106)),
        MRIToTensor2D(),
        transforms.Normalize(mean=[0.485], std=[0.229])  
    ])
    }


    image_path = os.path.join("/...", "LISA_challenge")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = MRIDataset(root_dir=os.path.join(image_path, train_file_name),attributes=attributes,transform=data_transform["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = MRIDataset(root_dir=os.path.join(image_path, val_file_name),attributes=attributes,transform=data_transform["val"])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultiOutputModel_decoder_new_loss(n_classes=3,device=device).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    savedir = "./parameter/MambaOut_muti_best.pth"

    n_train_samples = len(train_dataloader)

    print("Starting training ...")

    accuracy_mean = 0.0
    Tem_accuracy_noise = 0
    Tem_accuracy_zipper = 0
    Tem_accuracy_positioning = 0
    Tem_accuracy_banding = 0
    Tem_accuracy_motion = 0
    Tem_accuracy_contrast = 0
    Tem_accuracy_distortion = 0
    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy_noise = 0
        accuracy_zipper = 0
        accuracy_positioning = 0
        accuracy_banding = 0
        accuracy_motion = 0
        accuracy_contrast = 0
        accuracy_distortion = 0
        start_time = time.time()

        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_noise,batch_accuracy_zipper,batch_accuracy_positioning,batch_accuracy_banding,batch_accuracy_motion,batch_accuracy_contrast,batch_accuracy_distortion = \
                calculate_metrics(output, target_labels)

            accuracy_noise += batch_accuracy_noise
            accuracy_zipper += batch_accuracy_zipper
            accuracy_positioning += batch_accuracy_positioning
            accuracy_banding += batch_accuracy_banding
            accuracy_motion += batch_accuracy_motion
            accuracy_contrast += batch_accuracy_contrast
            accuracy_distortion += batch_accuracy_distortion


            loss_train.backward()
            optimizer.step()
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Time taken for one epoch: {epoch_time} seconds")
        print("epoch {:4d}, loss: {:.4f}, noise: {:.4f}, zipper: {:.4f}, positioning: {:.4f}, banding: {:.4f}, motion: {:.4f}, contrast: {:.4f}, distortion: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_noise / n_train_samples,
            accuracy_zipper / n_train_samples,
            accuracy_positioning / n_train_samples,
            accuracy_banding / n_train_samples,
            accuracy_motion / n_train_samples,
            accuracy_contrast / n_train_samples,
            accuracy_distortion / n_train_samples))

        tem_accuracy_mean,tem_accuracy_noise, tem_accuracy_zipper, tem_accuracy_positioning,tem_accuracy_banding,tem_accuracy_motion,tem_accuracy_contrast,tem_accuracy_distortion = validate(model, val_dataloader, epoch, device)
        if tem_accuracy_mean > accuracy_mean:
            accuracy_mean = tem_accuracy_mean
            Tem_accuracy_noise = tem_accuracy_noise
            Tem_accuracy_zipper = tem_accuracy_zipper
            Tem_accuracy_positioning = tem_accuracy_positioning
            Tem_accuracy_banding = tem_accuracy_banding
            Tem_accuracy_motion = tem_accuracy_motion
            Tem_accuracy_contrast = tem_accuracy_contrast
            Tem_accuracy_distortion = tem_accuracy_distortion
            checkpoint_save(model, savedir, epoch)


    print("Best accuracy mean:{:.4f}".format(accuracy_mean))
    print("Best accuracy all kinds, noise: {:.4f}, zipper: {:.4f}, positioning: {:.4f}, banding: {:.4f}, motion: {:.4f}, contrast: {:.4f}, distortion: {:.4f}".format(
        Tem_accuracy_noise,
        Tem_accuracy_zipper,
        Tem_accuracy_positioning,
        Tem_accuracy_banding,
        Tem_accuracy_motion,
        Tem_accuracy_contrast,
        Tem_accuracy_distortion))

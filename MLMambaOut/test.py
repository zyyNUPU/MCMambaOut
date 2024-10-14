import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from dataset import  AttributesDataset, MRIDataset_val,mean, std
from model import MultiOutputModel,MultiOutputModel_test,MultiOutputModel_decoder_new_loss_test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))


def validate(model, dataloader, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_noise = 0
        accuracy_zipper = 0
        accuracy_positioning = 0
        accuracy_banding = 0
        accuracy_motion = 0
        accuracy_contrast = 0
        accuracy_distortion = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_noise,batch_accuracy_zipper,batch_accuracy_positioning,batch_accuracy_banding,batch_accuracy_motion,batch_accuracy_contrast,batch_accuracy_distortion = \
                calculate_metrics(output, target_labels)

            accuracy_noise += batch_accuracy_noise
            accuracy_zipper += batch_accuracy_zipper
            accuracy_positioning += batch_accuracy_positioning
            accuracy_banding += batch_accuracy_banding
            accuracy_motion += batch_accuracy_motion
            accuracy_contrast += batch_accuracy_contrast
            accuracy_distortion += batch_accuracy_distortion

    n_samples = len(dataloader)
    avg_loss /= n_samples
    accuracy_noise /= n_samples
    accuracy_zipper /= n_samples
    accuracy_positioning /= n_samples
    accuracy_banding /= n_samples
    accuracy_motion /= n_samples
    accuracy_contrast /= n_samples
    accuracy_distortion /= n_samples
    print('-' * 72)
    print("Validation  loss: {:.4f}, noise: {:.4f}, zipper: {:.4f}, positioning: {:.4f}, banding: {:.4f}, motion: {:.4f}, contrast: {:.4f}, distortion: {:.4f}".format(
        avg_loss, accuracy_noise, accuracy_zipper, accuracy_positioning,accuracy_banding,accuracy_motion,accuracy_contrast,accuracy_distortion))

    model.train()
    accuracy_mean = (accuracy_noise+accuracy_zipper+accuracy_positioning+accuracy_banding+accuracy_motion+accuracy_contrast+accuracy_distortion)/7
    print("Accuracy mean: {:.4f}\n".format(accuracy_mean))
    return accuracy_mean,accuracy_noise, accuracy_zipper, accuracy_positioning,accuracy_banding,accuracy_motion,accuracy_contrast,accuracy_distortion

def visualize_grid(model, dataloader, attributes, device, csv_location=None, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []

    predicted_noise_all = []
    predicted_zipper_all = []
    predicted_positioning_all = []
    predicted_banding_all = []
    predicted_motion_all = []
    predicted_contrast_all = []
    predicted_distortion_all = []

    filenames = []

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            filename = batch['filename']

            output = model(img.to(device))

            # get the most confident prediction for each image
            _, predicted_noises = output['noise'].cpu().max(1)
            _, predicted_zippers = output['zipper'].cpu().max(1)
            _, predicted_positionings = output['positioning'].cpu().max(1)
            _, predicted_bandings = output['banding'].cpu().max(1)
            _, predicted_motions = output['motion'].cpu().max(1)
            _, predicted_contrasts = output['contrast'].cpu().max(1)
            _, predicted_distortions = output['distortion'].cpu().max(1)


            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_noise = attributes.Noise_id_to_name[predicted_noises[i].item()]
                predicted_zipper = attributes.Zipper_id_to_name[predicted_zippers[i].item()]
                predicted_positioning = attributes.Positioning_id_to_name[predicted_positionings[i].item()]
                predicted_banding = attributes.Banding_id_to_name[predicted_bandings[i].item()]
                predicted_motion = attributes.Motion_id_to_name[predicted_motions[i].item()]
                predicted_contrast = attributes.Contrast_id_to_name[predicted_contrasts[i].item()]
                predicted_distortion = attributes.Distortion_id_to_name[predicted_distortions[i].item()]
                

                predicted_noise_all.append(predicted_noise)
                predicted_zipper_all.append(predicted_zipper)
                predicted_positioning_all.append(predicted_positioning)
                predicted_banding_all.append(predicted_banding)
                predicted_motion_all.append(predicted_motion)
                predicted_contrast_all.append(predicted_contrast)
                predicted_distortion_all.append(predicted_distortion)

                filenames.append(filename[i])

                imgs.append(image)
                #labels.append("{}\n{}\n{}".format(predicted_noise, predicted_zipper, predicted_positioning))
        
        # Save results to CSV
        df = pd.DataFrame({
            'filename': filenames,
            'Noise': predicted_noise_all,
            'Zipper': predicted_zipper_all,
            'Positioning': predicted_positioning_all,
            'Banding': predicted_banding_all,
            'Motion': predicted_motion_all,
            'Contrast': predicted_contrast_all,
            'Distortion': predicted_distortion_all
        })
        patch_path = os.path.join(csv_location, "result_val_patch.csv") 
        df.to_csv(patch_path, index=False)
        print(f"Results path saved to {patch_path}")

        df = pd.read_csv(patch_path)

        df['Prefix'] = df['filename'].str.extract(r'(LISA_VALIDATION_\d{4}_LF_\w{3})')

        grouped = df.groupby('Prefix')[['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']].mean()


        output_df = grouped.reset_index()
        output_df['filename'] = output_df['Prefix'] + '.nii.gz'
        output_df = output_df.drop(columns='Prefix')

        columns_order = ['filename', 'Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
        output_df = output_df[columns_order]

        rounding_columns = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
        output_df[rounding_columns] = output_df[rounding_columns].round(0)

        output_df.to_csv(os.path.join(csv_location, "LISA_LF_QC_predictions.csv") , index=False)


    model.train()


def calculate_metrics(output, target):
    _, predicted_noise = output['noise'].cpu().max(1)
    gt_noise = target['noise_labels'].cpu()

    _, predicted_zipper = output['zipper'].cpu().max(1)
    gt_zipper = target['zipper_labels'].cpu()

    _, predicted_positioning = output['positioning'].cpu().max(1)
    gt_positioning = target['positioning_labels'].cpu()

    _, predicted_banding = output['banding'].cpu().max(1)
    gt_banding = target['banding_labels'].cpu()

    _, predicted_motion = output['motion'].cpu().max(1)
    gt_motion = target['motion_labels'].cpu()

    _, predicted_contrast = output['contrast'].cpu().max(1)
    gt_contrast = target['contrast_labels'].cpu()

    _, predicted_distortion = output['distortion'].cpu().max(1)
    gt_distortion = target['distortion_labels'].cpu()


    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        accuracy_noise = balanced_accuracy_score(y_true=gt_noise.numpy(), y_pred=predicted_noise.numpy())
        accuracy_zipper = balanced_accuracy_score(y_true=gt_zipper.numpy(), y_pred=predicted_zipper.numpy())
        accuracy_positioning = balanced_accuracy_score(y_true=gt_positioning.numpy(), y_pred=predicted_positioning.numpy())
        accuracy_banding = balanced_accuracy_score(y_true=gt_banding.numpy(), y_pred=predicted_banding.numpy())
        accuracy_motion = balanced_accuracy_score(y_true=gt_motion.numpy(), y_pred=predicted_motion.numpy())
        accuracy_contrast = balanced_accuracy_score(y_true=gt_contrast.numpy(), y_pred=predicted_contrast.numpy())
        accuracy_distortion = balanced_accuracy_score(y_true=gt_distortion.numpy(), y_pred=predicted_distortion.numpy())

    return accuracy_noise,accuracy_zipper,accuracy_positioning,accuracy_banding,accuracy_motion,accuracy_contrast,accuracy_distortion


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, default="./parameter/MambaOut_muti_best.pth", help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./LISA_LF_QC_updated.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()
    csv_location = "./"

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

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
    image_path = os.path.join("..", "..")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    test_file_name = ".."
    test_dataset = MRIDataset_val(root_dir=os.path.join(image_path, test_file_name),transform=data_transform["val"])
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel_decoder_new_loss_test(n_classes=3,device=device).to(device)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, attributes, device, csv_location=csv_location,checkpoint=args.checkpoint)

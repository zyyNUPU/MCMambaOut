import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import resnet2D
import mambaout
from ml_decoder import MLDecoder


class DynamicFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        prob = F.softmax(inputs, dim=1)
        p_t = torch.gather(prob, 1, targets.unsqueeze(1)).squeeze()
        alpha_t = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = alpha_t * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiOutputModel(nn.Module):
    def __init__(self, n_classes,device):
        super().__init__()

        #MambaOut
        self.base_model = mambaout.MambaOut(1,depths=[3, 4, 27, 3],dims=[96, 192, 384, 576])
        pre_param = '/home/zyy/Modules/Classification/Test5_resnet/pre_param/mambaout_small.pth'
        assert os.path.exists(pre_param), "weights file: '{}' not exist.".format(pre_param)
        weights_dict = torch.load(pre_param, map_location=device)
        conv1_weight = weights_dict['downsample_layers.0.conv1.weight']
        weights_dict['downsample_layers.0.conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(self.base_model.load_state_dict(weights_dict, strict=False))
        last_channel = 576
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


        # create separate classifiers for our outputs
        self.noise = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.zipper = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.positioning = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.banding = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.motion = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.contrast = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.distortion = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'noise': self.noise(x),
            'zipper': self.zipper(x),
            'positioning': self.positioning(x),
            'banding': self.banding(x),
            'motion': self.motion(x),
            'contrast': self.contrast(x),
            'distortion': self.distortion(x)
        }

    # def get_loss(self, net_output, ground_truth):
    #     noise_loss = F.cross_entropy(net_output['noise'], ground_truth['noise_labels'])
    #     zipper_loss = F.cross_entropy(net_output['zipper'], ground_truth['zipper_labels'])
    #     positioning_loss = F.cross_entropy(net_output['positioning'], ground_truth['positioning_labels'])
    #     banding_loss = F.cross_entropy(net_output['banding'], ground_truth['banding_labels'])
    #     motion_loss = F.cross_entropy(net_output['motion'], ground_truth['motion_labels'])
    #     contrast_loss = F.cross_entropy(net_output['contrast'], ground_truth['contrast_labels'])
    #     distortion_loss = F.cross_entropy(net_output['distortion'], ground_truth['distortion_labels'])
    #     loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
    #     return loss, {'nosie': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss,'banding':banding_loss,'motion':motion_loss,'contrast':contrast_loss,'distortion':distortion_loss}
    
    def get_loss(self, net_output, ground_truth):
        focal_loss = DynamicFocalLoss(alpha=0.25, gamma=2.0)
        
        noise_loss = focal_loss(net_output['noise'], ground_truth['noise_labels'])
        zipper_loss = focal_loss(net_output['zipper'], ground_truth['zipper_labels'])
        positioning_loss = focal_loss(net_output['positioning'], ground_truth['positioning_labels'])
        banding_loss = focal_loss(net_output['banding'], ground_truth['banding_labels'])
        motion_loss = focal_loss(net_output['motion'], ground_truth['motion_labels'])
        contrast_loss = focal_loss(net_output['contrast'], ground_truth['contrast_labels'])
        distortion_loss = focal_loss(net_output['distortion'], ground_truth['distortion_labels'])
        
        loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
        return loss, {'noise': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss, 'banding': banding_loss, 'motion': motion_loss, 'contrast': contrast_loss, 'distortion': distortion_loss}

class MultiOutputModel_decoder_new_loss(nn.Module):
    def __init__(self, n_classes,device):
        super().__init__()

        #MambaOut
        self.base_model = mambaout.MambaOut(1,depths=[3, 4, 27, 3],dims=[96, 192, 384, 576])
        pre_param = '/home/zyy/Modules/Classification/Test5_resnet/pre_param/mambaout_small.pth'
        assert os.path.exists(pre_param), "weights file: '{}' not exist.".format(pre_param)
        weights_dict = torch.load(pre_param, map_location=device)
        conv1_weight = weights_dict['downsample_layers.0.conv1.weight']
        weights_dict['downsample_layers.0.conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(self.base_model.load_state_dict(weights_dict, strict=False))
        last_channel = 576
        self.id = nn.Identity()
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))


        # create separate classifiers for our outputs
        self.noise = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.zipper = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.positioning = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.banding = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.motion = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.contrast = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.distortion = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.id(x)
        #x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        #x = torch.flatten(x, 1)

        return {
            'noise': self.noise(x),
            'zipper': self.zipper(x),
            'positioning': self.positioning(x),
            'banding': self.banding(x),
            'motion': self.motion(x),
            'contrast': self.contrast(x),
            'distortion': self.distortion(x)
        }

    # def get_loss(self, net_output, ground_truth):
    #     noise_loss = F.cross_entropy(net_output['noise'], ground_truth['noise_labels'])
    #     zipper_loss = F.cross_entropy(net_output['zipper'], ground_truth['zipper_labels'])
    #     positioning_loss = F.cross_entropy(net_output['positioning'], ground_truth['positioning_labels'])
    #     banding_loss = F.cross_entropy(net_output['banding'], ground_truth['banding_labels'])
    #     motion_loss = F.cross_entropy(net_output['motion'], ground_truth['motion_labels'])
    #     contrast_loss = F.cross_entropy(net_output['contrast'], ground_truth['contrast_labels'])
    #     distortion_loss = F.cross_entropy(net_output['distortion'], ground_truth['distortion_labels'])
    #     loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
    #     return loss, {'nosie': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss,'banding':banding_loss,'motion':motion_loss,'contrast':contrast_loss,'distortion':distortion_loss}
    
    def get_loss(self, net_output, ground_truth):
        focal_loss = DynamicFocalLoss(alpha=4, gamma=2.0,reduction="sum")
        
        noise_loss = focal_loss(net_output['noise'], ground_truth['noise_labels'])
        zipper_loss = focal_loss(net_output['zipper'], ground_truth['zipper_labels'])
        positioning_loss = focal_loss(net_output['positioning'], ground_truth['positioning_labels'])
        banding_loss = focal_loss(net_output['banding'], ground_truth['banding_labels'])
        motion_loss = focal_loss(net_output['motion'], ground_truth['motion_labels'])
        contrast_loss = focal_loss(net_output['contrast'], ground_truth['contrast_labels'])
        distortion_loss = focal_loss(net_output['distortion'], ground_truth['distortion_labels'])
        
        loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
        return loss, {'noise': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss, 'banding': banding_loss, 'motion': motion_loss, 'contrast': contrast_loss, 'distortion': distortion_loss}




class MultiOutputModel_decoder_old_loss(nn.Module):
    def __init__(self, n_classes,device):
        super().__init__()

        #MambaOut
        self.base_model = mambaout.MambaOut(1,depths=[3, 4, 27, 3],dims=[96, 192, 384, 576])
        pre_param = '/home/zyy/Modules/Classification/Test5_resnet/pre_param/mambaout_small.pth'
        assert os.path.exists(pre_param), "weights file: '{}' not exist.".format(pre_param)
        weights_dict = torch.load(pre_param, map_location=device)
        conv1_weight = weights_dict['downsample_layers.0.conv1.weight']
        weights_dict['downsample_layers.0.conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(self.base_model.load_state_dict(weights_dict, strict=False))
        last_channel = 576
        self.id = nn.Identity()
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))


        # create separate classifiers for our outputs
        self.noise = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.zipper = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.positioning = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.banding = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.motion = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.contrast = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.distortion = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.id(x)
        #x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        #x = torch.flatten(x, 1)

        return {
            'noise': self.noise(x),
            'zipper': self.zipper(x),
            'positioning': self.positioning(x),
            'banding': self.banding(x),
            'motion': self.motion(x),
            'contrast': self.contrast(x),
            'distortion': self.distortion(x)
        }

    def get_loss(self, net_output, ground_truth):
        noise_loss = F.cross_entropy(net_output['noise'], ground_truth['noise_labels'])
        zipper_loss = F.cross_entropy(net_output['zipper'], ground_truth['zipper_labels'])
        positioning_loss = F.cross_entropy(net_output['positioning'], ground_truth['positioning_labels'])
        banding_loss = F.cross_entropy(net_output['banding'], ground_truth['banding_labels'])
        motion_loss = F.cross_entropy(net_output['motion'], ground_truth['motion_labels'])
        contrast_loss = F.cross_entropy(net_output['contrast'], ground_truth['contrast_labels'])
        distortion_loss = F.cross_entropy(net_output['distortion'], ground_truth['distortion_labels'])
        loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
        return loss, {'nosie': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss,'banding':banding_loss,'motion':motion_loss,'contrast':contrast_loss,'distortion':distortion_loss}
    
    # def get_loss(self, net_output, ground_truth):
    #     focal_loss = DynamicFocalLoss(alpha=0.25, gamma=2.0)
        
    #     noise_loss = focal_loss(net_output['noise'], ground_truth['noise_labels'])
    #     zipper_loss = focal_loss(net_output['zipper'], ground_truth['zipper_labels'])
    #     positioning_loss = focal_loss(net_output['positioning'], ground_truth['positioning_labels'])
    #     banding_loss = focal_loss(net_output['banding'], ground_truth['banding_labels'])
    #     motion_loss = focal_loss(net_output['motion'], ground_truth['motion_labels'])
    #     contrast_loss = focal_loss(net_output['contrast'], ground_truth['contrast_labels'])
    #     distortion_loss = focal_loss(net_output['distortion'], ground_truth['distortion_labels'])
        
    #     loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
    #     return loss, {'noise': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss, 'banding': banding_loss, 'motion': motion_loss, 'contrast': contrast_loss, 'distortion': distortion_loss}



class MultiOutputModel_decoder_new_loss_test(nn.Module):
    def __init__(self, n_classes,device):
        super().__init__()

        #MambaOut
        self.base_model = mambaout.MambaOut(1,depths=[3, 4, 27, 3],dims=[96, 192, 384, 576])
        # pre_param = '/home/zyy/Modules/Classification/Test5_resnet/pre_param/mambaout_small.pth'
        # assert os.path.exists(pre_param), "weights file: '{}' not exist.".format(pre_param)
        # weights_dict = torch.load(pre_param, map_location=device)
        # conv1_weight = weights_dict['downsample_layers.0.conv1.weight']
        # weights_dict['downsample_layers.0.conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)
        # for k in list(weights_dict.keys()):
        #     if "head" in k:
        #         del weights_dict[k]
        # print(self.base_model.load_state_dict(weights_dict, strict=False))
        last_channel = 576
        self.id = nn.Identity()
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))


        # create separate classifiers for our outputs
        self.noise = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.zipper = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.positioning = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.banding = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.motion = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.contrast = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.distortion = nn.Sequential(
            MLDecoder(3,initial_num_features=last_channel)
            # nn.Dropout(p=0.2),
            # nn.Linear(in_features=last_channel, out_features=n_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.id(x)
        #x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        #x = torch.flatten(x, 1)

        return {
            'noise': self.noise(x),
            'zipper': self.zipper(x),
            'positioning': self.positioning(x),
            'banding': self.banding(x),
            'motion': self.motion(x),
            'contrast': self.contrast(x),
            'distortion': self.distortion(x)
        }

    # def get_loss(self, net_output, ground_truth):
    #     noise_loss = F.cross_entropy(net_output['noise'], ground_truth['noise_labels'])
    #     zipper_loss = F.cross_entropy(net_output['zipper'], ground_truth['zipper_labels'])
    #     positioning_loss = F.cross_entropy(net_output['positioning'], ground_truth['positioning_labels'])
    #     banding_loss = F.cross_entropy(net_output['banding'], ground_truth['banding_labels'])
    #     motion_loss = F.cross_entropy(net_output['motion'], ground_truth['motion_labels'])
    #     contrast_loss = F.cross_entropy(net_output['contrast'], ground_truth['contrast_labels'])
    #     distortion_loss = F.cross_entropy(net_output['distortion'], ground_truth['distortion_labels'])
    #     loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
    #     return loss, {'nosie': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss,'banding':banding_loss,'motion':motion_loss,'contrast':contrast_loss,'distortion':distortion_loss}
    
    def get_loss(self, net_output, ground_truth):
        focal_loss = DynamicFocalLoss(alpha=0.25, gamma=2.0)
        
        noise_loss = focal_loss(net_output['noise'], ground_truth['noise_labels'])
        zipper_loss = focal_loss(net_output['zipper'], ground_truth['zipper_labels'])
        positioning_loss = focal_loss(net_output['positioning'], ground_truth['positioning_labels'])
        banding_loss = focal_loss(net_output['banding'], ground_truth['banding_labels'])
        motion_loss = focal_loss(net_output['motion'], ground_truth['motion_labels'])
        contrast_loss = focal_loss(net_output['contrast'], ground_truth['contrast_labels'])
        distortion_loss = focal_loss(net_output['distortion'], ground_truth['distortion_labels'])
        
        loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
        return loss, {'noise': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss, 'banding': banding_loss, 'motion': motion_loss, 'contrast': contrast_loss, 'distortion': distortion_loss}




class MultiOutputModel_test(nn.Module):
    def __init__(self, n_classes,device):
        super().__init__()


        #MambaOut
        self.base_model = mambaout.MambaOut(1,depths=[3, 4, 27, 3],dims=[96, 192, 384, 576])
        # pre_param = '/home/zyy/Modules/Classification/Test5_resnet/pre_param/mambaout_small.pth'
        # assert os.path.exists(pre_param), "weights file: '{}' not exist.".format(pre_param)
        # weights_dict = torch.load(pre_param, map_location=device)
        # conv1_weight = weights_dict['downsample_layers.0.conv1.weight']
        # weights_dict['downsample_layers.0.conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)
        # for k in list(weights_dict.keys()):
        #     if "head" in k:
        #         del weights_dict[k]
        # print(self.base_model.load_state_dict(weights_dict, strict=False))
        last_channel = 576
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


        # create separate classifiers for our outputs
        self.noise = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.zipper = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.positioning = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.banding = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.motion = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.contrast = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )
        self.distortion = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'noise': self.noise(x),
            'zipper': self.zipper(x),
            'positioning': self.positioning(x),
            'banding': self.banding(x),
            'motion': self.motion(x),
            'contrast': self.contrast(x),
            'distortion': self.distortion(x)
        }

    # def get_loss(self, net_output, ground_truth):
    #     noise_loss = F.cross_entropy(net_output['noise'], ground_truth['noise_labels'])
    #     zipper_loss = F.cross_entropy(net_output['zipper'], ground_truth['zipper_labels'])
    #     positioning_loss = F.cross_entropy(net_output['positioning'], ground_truth['positioning_labels'])
    #     banding_loss = F.cross_entropy(net_output['banding'], ground_truth['banding_labels'])
    #     motion_loss = F.cross_entropy(net_output['motion'], ground_truth['motion_labels'])
    #     contrast_loss = F.cross_entropy(net_output['contrast'], ground_truth['contrast_labels'])
    #     distortion_loss = F.cross_entropy(net_output['distortion'], ground_truth['distortion_labels'])
    #     loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
    #     return loss, {'nosie': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss,'banding':banding_loss,'motion':motion_loss,'contrast':contrast_loss,'distortion':distortion_loss}

    def get_loss(self, net_output, ground_truth):
        focal_loss = DynamicFocalLoss(alpha=0.25, gamma=2.0)
        
        noise_loss = focal_loss(net_output['noise'], ground_truth['noise_labels'])
        zipper_loss = focal_loss(net_output['zipper'], ground_truth['zipper_labels'])
        positioning_loss = focal_loss(net_output['positioning'], ground_truth['positioning_labels'])
        banding_loss = focal_loss(net_output['banding'], ground_truth['banding_labels'])
        motion_loss = focal_loss(net_output['motion'], ground_truth['motion_labels'])
        contrast_loss = focal_loss(net_output['contrast'], ground_truth['contrast_labels'])
        distortion_loss = focal_loss(net_output['distortion'], ground_truth['distortion_labels'])
        
        loss = noise_loss + zipper_loss + positioning_loss + banding_loss + motion_loss + contrast_loss + distortion_loss
        return loss, {'noise': noise_loss, 'zipper': zipper_loss, 'positioning': positioning_loss, 'banding': banding_loss, 'motion': motion_loss, 'contrast': contrast_loss, 'distortion': distortion_loss}


# c = torch.ones(1,1,92,106)
# # # model = MambaOut(1,69)

# # # out = model(c)
# # # print(out.shape)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# c = c.to(device=device)
# net = MultiOutputModel_decoder_new_loss(n_classes=3,device=device).to(device)

# out = net(c)
# print(out)
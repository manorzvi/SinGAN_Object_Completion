import os
import shutil
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def create_names(image_name, name, amount):
    masks_names = [image_name[:-4] + '_{}_{}.jpg'.format(name, i) for i in range(amount)]
    return masks_names


def save_here(images, images_names):
    assert isinstance(images, torch.Tensor)
    assert len(images.shape) == 4
    for i, name in enumerate(images_names):
        image_t = Image.fromarray(images[i, :, :, :].squeeze().mul(255).byte().cpu().numpy())
        print('Save {} here... '.format(name), end=' ')
        image_t = image_t.save(name)
        print('Done.')


def save2drive(dir_path, images_names):
    assert os.path.exists(dir_path)
    for name in images_names:
        assert os.path.exists(name)
        name_dest = os.path.join(dir_path, name)
        print('Save {} to drive... '.format(name), end=' ')
        shutil.copyfile(name, name_dest)
        print('Done.')


def apply_mask(masks, image, names_to_save, device,
               to_save_in_drive=True, drive_path=None):
    assert isinstance(masks, torch.Tensor)
    assert len(masks.shape) == 4
    assert isinstance(image, torch.Tensor)
    assert masks.shape[0] == len(names_to_save)
    assert masks.shape[2] == image.shape[1] and masks.shape[3] == image.shape[2]
    assert (to_save_in_drive == True and drive_path != None) or (to_save_in_drive == False and drive_path == None)

    for i, name in enumerate(names_to_save):
        masked = torch.mul(image.to(device), masks[i, :, :, :])
        masked = Image.fromarray(masked.mul(255).permute(1, 2, 0).byte().cpu().numpy())
        print('Save {} here... '.format(name), end=' ')
        masked = masked.save(name)
        print('Done.')

    if to_save_in_drive:
        save2drive(dir_path=drive_path, images_names=names_to_save)


def apply_patch(mask, image, name_to_save, device,
                offset_h, offset_v,
                to_save_in_drive=True, drive_path=None):
    # TODO: add assertions later! (manorz, 12/02/19)
    # assert isinstance(mask, torch.Tensor)
    # assert len(mask.shape) == 3
    # assert isinstance(image, torch.Tensor)
    # assert mask.shape[2] == image.shape[1] and mask.shape[3] == image.shape[2]
    # assert (to_save_in_drive == True and drive_path != None) or (to_save_in_drive == False and drive_path == None)

    masked = torch.mul(image.to(device), mask)
    masked_pil = Image.fromarray(masked.mul(255).permute(1, 2, 0).byte().cpu().numpy())
    _, argz_v, argz_h = np.where(mask.cpu().numpy() == 0)
    # TODO: add support for list of offsets. (manorz, 12/02/19)
    masked[:, argz_v, argz_h] = masked[:, argz_v + offset_v, argz_h + offset_h]
    new_masked_pil = Image.fromarray(masked.mul(255).permute(1, 2, 0).byte().cpu().numpy())

    print('Save {} here... '.format(name_to_save), end=' ')
    masked = new_masked_pil.save(name_to_save)
    print('Done.')

    print(drive_path, name_to_save)

    if to_save_in_drive:
        save2drive(dir_path=drive_path, images_names=name_to_save)













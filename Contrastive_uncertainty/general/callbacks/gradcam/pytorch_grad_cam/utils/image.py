import cv2
import numpy as np
import torch
from torch.distributions.transforms import Transform
from torchvision.transforms import Compose, Normalize, ToTensor


def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

'''
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
'''

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      mean,
                      std,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      ) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    #invert_mean, invert_std = -mean, 1/std
    # Based on https://github.com/pytorch/vision/issues/528 
    collated_cams = []
    mean, std = torch.tensor(mean), torch.tensor(std)
    
    preprocessing = Compose([
        ToTensor(),
        Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    ])
    for datapoint, mask_datapoint in zip(img,mask): # go through each individual image
        datapoint,mask_datapoint = preprocessing(datapoint), preprocessing(mask_datapoint) # torch tensor which is renormalized
        datapoint,mask_datapoint = datapoint.data.cpu().numpy(), mask_datapoint.data.cpu().numpy() # shape (channel, height, width)
        datapoint, mask_datapoint = datapoint.reshape(datapoint.shape[1],datapoint.shape[2],datapoint.shape[0]), mask_datapoint.reshape(mask_datapoint.shape[1],mask_datapoint.shape[2],mask_datapoint.shape[0])
        
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_datapoint), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        
        if np.max(datapoint) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")
        
        cam = heatmap + datapoint
        cam = cam / np.max(cam)
        collated_cams.append(np.uint8(255 * cam))
    return np.array(collated_cams)


    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

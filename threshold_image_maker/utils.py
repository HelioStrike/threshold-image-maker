import numpy as np
import cv2

def cleanImage(img):
    img = removeSmallConnectedComponents(img)
    return img

def removeSmallConnectedComponents(img, max_size=10):
    img = 255 - img
    if img.shape[-1] == 1:
        img = (img.reshape(*img.shape, 1)*255).astype(np.uint8)
    else:
        img = (img.reshape(*img.shape, 1)*255).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= max_size:
            img2[output == i + 1] = 1
    img2 = (img2*255).astype(np.uint8)
    img2 = 255 - img2
    return img2

def add_alpha_channel(img):
    channel = (img[:,:,0]==0).astype(np.uint8).reshape(*img.shape[:-1], 1)
    img = np.concatenate([img, channel*255], axis=-1)
    return img
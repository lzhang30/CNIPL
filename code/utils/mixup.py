# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
from scipy.special import erfinv

from scipy.ndimage.filters import gaussian_filter


def generate_cutmix_mask(shape, prop_range = 0.2, n_holes=1, random_aspect_ratio=True, within_bounds=True):
    if isinstance(prop_range, float):
        prop_range = (prop_range, prop_range)

    n_masks, _, h, w = list(shape)


    # mask = np.ones((h, w), np.float32)
    # valid = np.zeros((h ,w),np.float32)

    mask_props = np.random.uniform(prop_range[0], prop_range[1], size=(n_masks, n_holes))
    if random_aspect_ratio:
        y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(n_masks, n_holes)) * np.log(mask_props))
        x_props = mask_props / y_props
    else:
        y_props = x_props = np.sqrt(mask_props)

    fac = np.sqrt(1.0 / n_holes)
    y_props *= fac
    x_props *= fac

    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :])

    if within_bounds:
        positions = np.round((np.array((h, w)) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=2)
    else:
        centres = np.round(np.array((h, w)) * np.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

    masks = np.zeros(shape)
    for i, sample_rectangles in enumerate(rectangles):
        for y0, x0, y1, x1 in sample_rectangles:
            # print('len:', y0 - y1)
            # print('hig:', x0 - x1)

            masks[i,:, int(y0):int(y1), int(x0):int(x1)] = 1

    masks = torch.from_numpy(masks)

    return masks


def generate_cutout_mask(img_size):

    cutout_area = img_size[0] * img_size[1] * 0.2

    w = np.random.randint(img_size[1] / 2, img_size[1] + 1)
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

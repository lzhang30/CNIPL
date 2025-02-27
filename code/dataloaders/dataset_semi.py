import itertools
import os
import random
import re
from glob import glob
from scipy.ndimage import gaussian_filter
import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.nn as nn
from PIL import Image

from skimage import exposure
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
try:
    from ctaugment import OPS
except:
    from augmentations.ctaugment import OPS

import augmentations
from torchvision import transforms
from dataloaders.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, num=4, labeled_type="labeled", split='train', transform=None, fold="fold1", sup_type="scribble", ratio = 8):
        
        self._base_dir = base_dir

        if 'ACDC' in base_dir:
            self.sample_list = []
            self.split = split
            self.sup_type = sup_type
            print(f'label tyoe {sup_type}')
            self.transform = transform
            self.num = num
            self.labeled_type = labeled_type
            train_ids, test_ids = self._get_fold_ids(fold)
            if ratio == 8:
                all_labeled_ids = ["patient{:0>3}".format(
                    10 * i) for i in range(1, 11)]
            elif ratio == 4:
                all_labeled_ids = ["patient{:0>3}".format(
                    20 * i) for i in range(1, 6)]
            elif ratio == 16:
                all_labeled_ids = ["patient{:0>3}".format(
                    5 * i) for i in range(1, 21)]
            elif ratio == 25:
                all_labeled_ids = ["patient{:0>3}".format(
                    4 * i) for i in range(1, 25)]
            elif ratio == 100:
                all_labeled_ids = ["patient{:0>3}".format(i) for i in range(0, 80)]
                
            if self.split == 'train':
                
                self.all_slices = os.listdir(
                    self._base_dir + "/ACDC_training_slices")
                self.sample_list = []
                labeled_ids = [i for i in all_labeled_ids if i in train_ids]
                unlabeled_ids = [i for i in train_ids if i not in labeled_ids]
                if self.labeled_type == "labeled":
                    print("Labeled patients IDs", labeled_ids)
                    for ids in labeled_ids:
                        new_data_list = list(filter(lambda x: re.match(
                            '{}.*'.format(ids), x) != None, self.all_slices))
                        self.sample_list.extend(new_data_list)
                    print("total labeled {} samples".format(len(self.sample_list)))
                else:
                    print("Unlabeled patients IDs", unlabeled_ids)
                    for ids in unlabeled_ids:
                        new_data_list = list(filter(lambda x: re.match(
                            '{}.*'.format(ids), x) != None, self.all_slices))
                        self.sample_list.extend(new_data_list)
                    print("total unlabeled {} samples".format(len(self.sample_list)))

            elif self.split in ['val','test']:
                self.all_volumes = os.listdir(
                    self._base_dir + "/ACDC_training_volumes")
                self.sample_list = []
                for ids in test_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_volumes))
                    
                    self.sample_list.extend(new_data_list)
                    

                    
        elif 'WORD' in base_dir:
            self.sample_list = []
            self.split = split
            self.sup_type = sup_type
            print(f'label type {sup_type}')
            self.transform = transform
            self.num = num
            self.labeled_type = labeled_type
            
            #self.all_volumes = sorted(os.listdir(self._base_dir + "/train_volumes"))
            
            
            train_ids = sorted(os.listdir(self._base_dir + "/train_volumes"))
            #train_ids = sorted(train_ids)
            val_ids = sorted(os.listdir(self._base_dir + "/val_volumes"))
            test_ids = sorted(os.listdir(self._base_dir + "/test_volumes"))
            
            
            if fold == 'fold1':
                all_labeled_ids = train_ids[1::ratio]
            elif fold == 'fold2':
                all_labeled_ids = train_ids[2::ratio]
            else:
                all_labeled_ids = train_ids[3::ratio]
                
            all_unlabeled_ids = [id_ for id_ in train_ids if id_ not in all_labeled_ids]
            self.all_slices = os.listdir(self._base_dir + "/all_slices")
            if self.split == 'train':
                                
                #train_ids = sorted(os.listdir(self._base_dir + "/train_volumes"))
                #self.all_slices = os.listdir(self._base_dir + "/all_slices")
                self.sample_list = []

                if self.labeled_type == "labeled":
                    print("Labeled patients IDs", all_labeled_ids)
                    
                    
                    for ids in all_labeled_ids:
                        #print(ids)
                        new_data_list = list(filter(lambda x: re.match(
                            '{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_slices))
                        self.sample_list.extend(new_data_list)
                        #print([ids,new_data_list])
                    print("total labeled {} samples".format(len(self.sample_list)))
                    
                else:
                    print("Unlabeled patients IDs", all_unlabeled_ids)
                    for ids in all_unlabeled_ids:
                        new_data_list = list(filter(lambda x: re.match(
                            '{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_slices))
                        self.sample_list.extend(new_data_list)
                    print("total unlabeled {} samples".format(len(self.sample_list)))
            
            elif self.split == 'val':
                print('val set loaded')
                #self.all_slices = os.listdir(self._base_dir + "/all_slices")
                self.val_ids = sorted(os.listdir(self._base_dir + "/val_volumes"))
                for ids in self.val_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids.replace(".h5", "")), x) != None, self.val_ids))
                    self.sample_list.extend(new_data_list)
                
            elif self.split == 'test':
                print('test set loaded')
                #self.all_volumes = sorted(os.listdir(self._base_dir + "/all_volumes"))[:-20]
                self.val_ids = sorted(os.listdir(self._base_dir + "/test_volumes"))
                for ids in self.val_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids.replace(".h5", "")), x) != None, self.val_ids))
                    self.sample_list.extend(new_data_list)



            else:
                raise ValueError("Unknown data types")
            
        elif 'MSCMR' in base_dir:
            #training_lsit = [2, 4, 6, 7, 9, 13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 31, 32, 34, 37, 39, 42, 44, 45]
            #test_list = [3,5,10,11,12,16,17,23,28,30,33,35,38,40,43]
            self.sample_list = []
            self.split = split
            self.sup_type = sup_type
            print(f'label type {sup_type}')
            self.transform = transform
            self.num = num
            self.labeled_type = labeled_type
            
            #self.all_volumes = sorted(os.listdir(self._base_dir + "/train_volumes"))
            
            
            train_ids = [str(poi) for poi in [2, 4, 6, 7, 9, 13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 31, 32, 34, 37, 39, 42, 44, 45]]
            #train_ids = sorted(train_ids)
            val_ids = sorted(os.listdir(self._base_dir + "/MSCMR_training_volumes"))
            test_ids = sorted(os.listdir(self._base_dir + "/MSCMR_testing_volumes"))
            ratio = len(train_ids) // 5

                        
            if fold == 'fold1':
                all_labeled_ids = train_ids[1::ratio]
            elif fold == 'fold2':
                all_labeled_ids = train_ids[2::ratio]
            else:
                all_labeled_ids = train_ids[3::ratio]

            all_unlabeled_ids = [id_ for id_ in train_ids if id_ not in all_labeled_ids]
            self.all_slices = os.listdir(self._base_dir + "/MSCMR_training_slices")

            if self.split == 'train':
                                
                #train_ids = sorted(os.listdir(self._base_dir + "/train_volumes"))
                #self.all_slices = os.listdir(self._base_dir + "/all_slices")
                self.sample_list = []

                if self.labeled_type == "labeled":
                    print("Labeled patients IDs", all_labeled_ids)
                    
                    
                    for ids in all_labeled_ids:
                        #print(ids)
                        new_data_list = list(filter(lambda x: re.match(
                            'subject{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_slices))
                        self.sample_list.extend(new_data_list)
                        #print([ids,new_data_list])
                    print("total labeled {} samples".format(len(self.sample_list)))
                    
                else:
                    print("Unlabeled patients IDs", all_unlabeled_ids)
                    for ids in all_unlabeled_ids:
                        new_data_list = list(filter(lambda x: re.match(
                            'subject{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_slices))
                        self.sample_list.extend(new_data_list)
                    print("total unlabeled {} samples".format(len(self.sample_list)))
            
            elif self.split == 'val':
                print('val set loaded')
                #self.all_slices = os.listdir(self._base_dir + "/all_slices")
                self.val_ids = sorted(os.listdir(self._base_dir + "/MSCMR_training_volumes"))
                self.sample_list  = self.val_ids
                '''
                print()
                for ids in self.val_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        'subject{}.*'.format(ids.replace(".h5", "")), x) != None, self.val_ids))
                    self.sample_list.extend(new_data_list)
                '''
            elif self.split == 'test':
                print('test set loaded')
                #self.all_volumes = sorted(os.listdir(self._base_dir + "/all_volumes"))[:-20]
                self.val_ids = sorted(os.listdir(self._base_dir + "/MSCMR_testing_volumes"))
                self.sample_list  = self.val_ids
                '''
                for ids in self.val_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        'subject{}.*'.format(ids.replace(".h5", "")), x) != None, self.val_ids))
                '''
            else:
                raise ValueError("Unknown data types")


            # if num is not None and self.split == "train":
            #     self.sample_list = self.sample_list[:num]
        else:
            raise ValueError("Unknown dataset")




    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if 'ACDC' in self._base_dir:
            if self.split == "train":
                h5f = h5py.File(self._base_dir +
                                "/ACDC_training_slices/{}".format(case), 'r')
            else:
                h5f = h5py.File(self._base_dir +
                                "/ACDC_training_volumes/{}".format(case), 'r')
        elif 'WORD' in self._base_dir:
            if self.split == "train":
                h5f = h5py.File(self._base_dir +
                                "/all_slices/{}".format(case), 'r')
            elif self.split == "val":
                h5f = h5py.File(self._base_dir +
                                "/val_volumes/{}".format(case), 'r')
            else:
                h5f = h5py.File(self._base_dir +
                            "/test_volumes/{}".format(case), 'r')   
        else:
            if self.split == "train":
                h5f = h5py.File(self._base_dir +
                                "/MSCMR_training_slices/{}".format(case), 'r')
            elif self.split == "val":
                h5f = h5py.File(self._base_dir +
                                "/MSCMR_training_volumes/{}".format(case), 'r')
            else:
                h5f = h5py.File(self._base_dir +
                            "/MSCMR_testing_volumes/{}".format(case), 'r')   

                
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            label = h5f[self.sup_type][:]
            #print(self.sup_type)
            #print(h5f[self.sup_type][:].max())
            if 'WORD' in self._base_dir:  
                mask = label[:] == 255
                if 'select' in self._base_dir:
                    label[mask] = 9

                else:
                    label[mask] = 17
            sample = {'image': image, 'label': label}
            
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            if 'WORD' in self._base_dir:  
                mask = label[:] == 255
                label[mask] = 17
                image = torch.from_numpy(image.astype(np.float32))
                label = torch.from_numpy(label.astype(np.uint8))
                sample = {'image': image, 'label': label}

            elif 'MSCMR' in self._base_dir:
                image = torch.from_numpy(image.astype(np.float32))
                label = torch.from_numpy(label.astype(np.uint8))
                sample = {'image': image, 'label': label}
            else:
                sample = {'image': image, 'label': label}
        sample["idx"] = case.split("_")[0]
        return sample






class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)



    
def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = gaussian_filter(img, sigma=sigma)
    return img

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


def random_gamma_correction(image, gamma_range=(0.8, 1.2)):
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    return exposure.adjust_gamma(image, gamma)

def random_contrast_adjustment(image, contrast_range=(0.8, 1.2)):
    v_min, v_max = np.percentile(image, (0.2, 99.8))
    image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    factor = random.uniform(contrast_range[0], contrast_range[1])
    mean = np.mean(image)
    return mean + factor * (image - mean)






class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=label.max())
            else:
                image, label = random_rotate(image, label, cval=label.max())
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample
    
    
    
import copy
class RandomGenerator_CCM(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=label.max())
        
        raw = copy.deepcopy(image)
        
        image, label = random_noise(image, label)

        x, y = raw.shape
        image_w = copy.deepcopy(image)
        image_w = zoom(
            image_w, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        raw = zoom(
            raw, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        a,b,c = random.random(),random.random(),random.random()
        if a <= 0.5:
            image, label = random_rescale_intensity(image, label)
        elif b <= 0.5:
            image = blur(image, p=1)
        else:
            image, label = random_equalize_hist(image, label)
            

        
        image_s = image
        image_s = zoom(
            image_s, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_w = torch.from_numpy(
            image_w.astype(np.float32)).unsqueeze(0)
        image_s = torch.from_numpy(
            image_s.astype(np.float32)).unsqueeze(0)
        
        raw = torch.from_numpy(
            raw.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'raw':raw,'image_w': image_w, 'image_s': image_s, 'label': label}
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


'''
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model
'''
import copy

def cutmix(image1, image2, label1, label2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    h, w = image1.shape
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    
    # Randomly choose the top left corner
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    # Create mixed images and labels
    image1[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
    label1[:, y1:y2, x1:x2] = label2[:, y1:y2, x1:x2]

    return image1, label1

class RandomGenerator_Strong_Weak(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=label.max())
        #if random.random() > 0.5:
        #   image, label = random_noise(image, label)
        

        x, y = image.shape
        image_w = copy.deepcopy(image)
        image_w = zoom(
            image_w, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        a,b,c = random.random(),random.random(),random.random()
        
        if a <= 0.5:
            image = random_gamma_correction(image)
        if b <= 0.5:
            image= random_contrast_adjustment(image)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
            
        image_s = image
        image_s = zoom(
            image_s, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_w = torch.from_numpy(
            image_w.astype(np.float32)).unsqueeze(0)
        image_s = torch.from_numpy(
            image_s.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'image_w': image_w, 'image_s': image_s, 'label': label}
        return sample



class RandomGenerator_Strong_Weak_ours(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=label.max())
        #if random.random() > 0.5:
        #   image, label = random_noise(image, label)
        

        x, y = image.shape
        image_w = copy.deepcopy(image)
        image_w = zoom(
            image_w, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        a,b,c = random.random(),random.random(),random.random()
        

        image, label = random_noise(image, label)
            
        image_s = image
        image_s = zoom(
            image_s, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_w = torch.from_numpy(
            image_w.astype(np.float32)).unsqueeze(0)
        image_s = torch.from_numpy(
            image_s.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'image_w': image_w, 'image_s': image_s, 'label': label}
        return sample



    
def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class RandomGenerator_Strong_Weak_uni(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if random.random() > 0.5:
            image, label = random_flip(image, label)


        x, y = image.shape
        img = copy.deepcopy(image)
        img = zoom(
            img, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        mask = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1, img_s2 = copy.deepcopy(img), copy.deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
        mask = torch.from_numpy(mask.astype(np.float)).unsqueeze(0).long()
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        #cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        #cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        

        sample = {'image_w': img, 'image_s': img_s1, 'image_s2': img_s2, 'label': mask}
        return sample
    







def random_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]),
                    -2 * sigma, 2 * sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, label, prob=0.5):
    if random.random() >= prob:
        return x, label
    points = [[0, 0], [random.random(), random.random()], [
        random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label


def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label

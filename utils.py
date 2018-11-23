import os
import numpy as np
from PIL import Image, ImageOps
import PIL



def get_data(path):
    imgfolders = get_imgfolders(path)
    img_dict = get_imgdict(imgfolders)
    labels = get_labels(imgfolders, img_dict)
    train_data, train_label, test_data, test_label = give_tr_test(img_dict, imgfolders, labels)
    train_perm , test_perm = np.random.permutation(train_data.shape[0]), np.random.permutation(test_data.shape[0])
    train_data, train_label = train_data[train_perm], train_label[train_perm]
    test_data, test_label = test_data[test_perm], test_label[test_perm]
    return train_data, train_label, test_data, test_label

def get_batch(flist, mean):
    batch = []
    for impath in flist:
        img = imread(impath)
        img = crop_img(img, mean)
        batch.append(img)
    return np.asarray(batch)

def get_imgfolders(path):
    fnames = os.listdir(path)
    imgfolders = []
    for fname in fnames:
       imgfolders.append(os.path.join(path, fname))
    return imgfolders

def imread(path):
    img = Image.open(path)
    return img


def get_imgdict(imgfolders):
    img_dict ={}
    for folder in imgfolders:
        imgs = join_path(folder, os.listdir(folder))
        img_dict[folder] = imgs
    return img_dict

def get_labels(imgfolders, img_dict):
    label_nmb = 0
    labels = {}
    for folder in imgfolders:
        imgs = img_dict[folder]
        labels[folder] = assign_label(len(imgs), label_nmb)
        label_nmb += 1
    return labels

def give_tr_test(img_dict,folders, labels):
    nmb_of_test_imgs = 15
    train_data, test_data, train_label, test_label = [], [], [], []
    for folder in folders:
        train_end = len(img_dict[folder]) - nmb_of_test_imgs
        train_data = np.concatenate((train_data,img_dict[folder][: train_end]))
        test_data = np.concatenate((test_data, img_dict[folder][train_end:]))
        train_label = np.concatenate((train_label, labels[folder][:train_end]))
        test_label = np.concatenate((test_label, labels[folder][train_end :]))
    return train_data, train_label, test_data, test_label


def join_path(path, folders):
    joined = []
    for name in folders:
        joined.append(os.path.join(path, name))
    return joined

def assign_label(size, label):
    return label * np.ones(size).astype(np.int32)

def check_path(path):
    if os.path.exists(path):
        return path
    else:
        raise OSError('Path not Found')

def check_batch(batch_size):
    if isinstance(batch_size, int) and batch_size > 0:
        return batch_size
    else:
        raise TypeError()

def crop_img(img, mean, size=224):
    short_axis = np.min(img.size)
    ratio = size / short_axis
    new_size = (ratio * img.size[0], ratio * img.size[1])
    img.thumbnail(new_size)
    cropped = ImageOps.fit(img, (size, size))
    cropped -= mean
    return np.array(cropped)


def separate_validation(train_data, chunk_num, val_index):
    size = len(train_data)
    chunk_size = size // chunk_num
    last_val_idx = val_index * chunk_size
    first_val_idx = last_val_idx - chunk_size 
    print(first_val_idx, last_val_idx, size, chunk_size)
    validation_set = np.copy(train_data[first_val_idx: last_val_idx])
    print(validation_set.shape)
    train_set = np.delete(train_data, range(first_val_idx, last_val_idx))
    return train_set, validation_set








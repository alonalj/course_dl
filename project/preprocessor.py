'''
# Data prep
dataset augmentation:
(1) crop images (into smaller crops only to not include hints (e.g. black edges) of which tile this is
(2) all other relevant augmentations we saw in the code provided for transfer learning in ex. 3
for image,
    for augmentation of image:
        shred in preprocessor into subfolders with labels in name ( check keras input?)
for each folder randomly select t ood tiles from another folder and add them
split folders into train, val, test parent-folders randomly
'''

import os
import pickle as pkl
import numpy as np
from conf import Conf
import random


def _shredder(raw_input_dir, data_type, c, output_dir):
    import cv2
    import os

    Xa = []
    Xb = []
    y = []

    # raw_input_dir = "images/"

    files = os.listdir(raw_input_dir)
    files_dict = load_obj(c.data_split_dict)
    files = files_dict[data_type]  # augment only train files
    files = files
    # files = [f for f in files if "n01440764_7267" in f]
    list_of_folders = []
    # update this number for 4X4 crop 2X2 or 5X5 crops.
    # tiles_per_dim = 4
    if data_type == 'train':
        crop_start_w = range(0, 46, 15)
        crop_start_h = range(0, 46, 15)
        reshape_options = [True, False]
    else:
        crop_start_w = [0]
        crop_start_h = [0]
        reshape_options = [False]

    for reshape in reshape_options:
        for c_w in crop_start_w:
            for c_h in crop_start_h:
                crops_previous = []
                names_previous = []

                for f in files:
                    # TODO: add another for loop to do the same also for augmented tiles, but make sure OoD is from a previous image, not from a previous augmentation
                    if c_w == 0 and c_h == 0 and reshape:
                        continue
                    filename = f.split('.')[0]+'_crw_'+str(c_w)+'_crh_'+str(c_h)+'_reshape_'+str(reshape)
                    folder_name = output_dir + filename
                    folder_output_dir = folder_name + '/'
                    list_of_folders.append(filename)
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    os.mkdir(folder_output_dir)

                    # add OOD from previous crops if exist:
                    if len(crops_previous) > 0:
                        # some of the time crate empty image (we will use this in case len(OoD) < t)
                        idx = np.arange(len(crops_previous))
                        np.random.shuffle(idx)
                        for c_idx in idx[:c.tiles_per_dim]:
                            if random.random() < 0.1:
                                null_img = np.zeros((c.max_size, c.max_size))
                                cv2.imwrite(folder_output_dir + names_previous[c_idx], null_img)
                            else:
                                cv2.imwrite(folder_output_dir+names_previous[c_idx], crops_previous[c_idx])
                    else:
                        folder_output_dir_first_img = folder_output_dir  # first img has no "previous", will use last img later

                    im = cv2.imread(raw_input_dir + f)
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    original_height, original_width = im.shape[0], im.shape[1]
                    # augmentation - crop image  # TODO: remove this for ducments - probably easier due to white gap naturally present on page

                    im = im[c_w:im.shape[0]-c_w, c_h:im.shape[1]-c_h]
                        # cv2.imshow("cropped", cropped)

                    height = im.shape[0]
                    width = im.shape[1]
                    if original_height != height and original_width != width and reshape:
                        try:
                            im = cv2.resize(im,(original_height, original_width))
                        except:
                            continue

                    frac_h = height // c.tiles_per_dim
                    frac_w = width // c.tiles_per_dim
                    i = 0
                    crops_previous = []
                    names_previous = []
                    for h in range(c.tiles_per_dim):
                        for w in range(c.tiles_per_dim):
                            crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                            cv2.imwrite(folder_output_dir + f[:-4] + "_{}.jpg".format(str(i).zfill(2)), crop)
                            crops_previous.append(crop)
                            names_previous.append(f[:-4] + "_{}_{}.jpg".format(str(i).zfill(2), str(-1))) # -1 indicates it will OoD for next image
                            i = i + 1

                    # take OoD from last image in files as "previous" for first img

                    if f == files[-1]:
                        idx = np.arange(len(crops_previous))
                        np.random.shuffle(idx)
                        for c_idx in idx[:c.tiles_per_dim]:
                            cv2.imwrite(folder_output_dir_first_img + names_previous[c_idx], crops_previous[c_idx])

    files_dict[data_type] = list_of_folders
    print(files_dict)
    save_obj(files_dict, c.data_split_dict)


def split_into_train_val_test(dataset_folder, portion_train, portion_val, dict_name):
    train_test_val_dict = {}
    folders_original = os.listdir(dataset_folder)
    folders = list(set([folders_original[i].split('_cr')[0] for i in range(len(folders_original))]))
    folders = [f for f in folders if 'DS_Store' not in f]
    np.random.shuffle(folders)
    num_folders = len(folders)
    stop_idx_train = int(num_folders*portion_train)
    stop_idx_val = stop_idx_train+int(num_folders*portion_val)
    train_folders = folders[:stop_idx_train]
    val_folders = folders[stop_idx_train : stop_idx_val]
    test_folders = folders[stop_idx_val :]

    train_folders = [f for f in folders_original if f.split('_cr')[0] in train_folders]
    val_folders = [f for f in folders_original if f.split('_cr')[0] in val_folders]
    test_folders = [f for f in folders_original if f.split('_cr')[0] in test_folders]

    train_test_val_dict['train'] = train_folders
    train_test_val_dict['val'] = val_folders
    train_test_val_dict['test'] = test_folders
    save_obj(train_test_val_dict, dict_name)


def save_obj(obj, name, directory=''):
    with open(directory + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def load_obj(name, directory=''):
    with open(directory + name + '.pkl', 'rb') as f:
        return pkl.load(f)


def resize_image(image, max_size=None, resize_factor=None, simple_reshape=True):
    import cv2
    import math
    if simple_reshape:
        im_resized = cv2.resize(image, (max_size, max_size))
    else:
        if resize_factor and not max_size:
            im_resized = cv2.resize(image, (image.shape[0] // resize_factor, image.shape[1] // resize_factor))  # TODO: verify indicese don't need swapping...
        elif max_size and not resize_factor:
            if image.shape[0] > image.shape[1]:
                ratio = math.ceil(image.shape[0] / float(max_size))
            else:
                ratio = math.ceil(image.shape[1] / float(max_size))
            im_resized = cv2.resize(image, (image.shape[0] // ratio, image.shape[1] // ratio))  # TODO: verify indicese don't need swapping...
        else:
            raise Exception("One, and only one, of max_size and resize_factor should be defined.")

        # TODO: because of OoD as well as images being of different size, need to do zero pad now
        # img = cv2.imread("img_src.jpg")
        shape = im_resized.shape
        w = shape[1]
        h = shape[0]
        slack_w = max_size - w  # padding size w
        slack_h = max_size - h  # padding size h
        # to avoid always padding img the same way, we randomly choose where img is located within the padding limits, also
        # a means of data augmentation to increase train size. For test time this is not very important... TODO: verify (try running with reshaping image, no padding)
        import random
        start_w = 0 #random.randint(0, slack_w)
        end_w = start_w + w
        start_h = 0 #random.randint(0, slack_h)
        end_h = start_h + h
        base_size = max_size, max_size
        base = np.zeros(base_size, dtype=np.uint8)
        # cv2.rectangle(base, (0, 0), (max_size, max_size), (255, 255))
        base[start_h:end_h, start_w:end_w] = im_resized
        im_resized = base
    return im_resized / 255.


def run_shredder(c):
    if 'doc' in c.data_split_dict:
        output_dir = "dataset_{}_isImg_False/".format(c.tiles_per_dim)
    else:
        output_dir = "dataset_{}_isImg_True/".format(c.tiles_per_dim)
    if os.path.exists(output_dir):
        print("Already shredded for: isImg {} and n_tiles_per_dim {}".format(c.is_images, c.tiles_per_dim))
        return
    dict_name = c.data_split_dict
    if 'doc' in c.data_split_dict:
        split_into_train_val_test('documents', 0.75, 0.15, dict_name)
    else:
        split_into_train_val_test('images', 0.75, 0.15, dict_name)
    d = load_obj(dict_name)
    # TODO: need this for multiple tile sizes, as well as for documents
    for data_type in ['train', 'val', 'test']:
        print("Shredding for {}".format(data_type))
        if 'doc' in c.data_split_dict:
            _shredder("documents/", data_type, c, output_dir)
        else:
            _shredder("images/", data_type, c, output_dir)


# if __name__ == '__main__':
    # # print(len(d['train']))
    # print(d['val'])
    # c = Conf()
    # for is_images in [True, False]:
        # c.data_split_dict = "train_test_val_dict_isImg_{}".format(str(is_images))

    # dict_name = 'train_test_val_dict_doc'
    # split_into_train_val_test('documents', 0.75, 0.15, dict_name)
    # d = load_obj(dict_name)
    # for data_type in ['train', 'val', 'test']:
    #     shredder("documents/", 2, data_type, dict_name)
    #     # shredder("documents/", 4)
    #     # shredder("documents/", 5)


    # import keras
    # print(keras.__version__)


# datagen = keras.preprocessing.ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
#
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# train_generator = datagen.flow_from_directory(
#         'dataset',
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
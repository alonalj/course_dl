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

def shredder(raw_input_dir, tiles_per_dim):
    import cv2
    import os

    Xa = []
    Xb = []
    y = []

    crops_previous = []
    names_previous = []

    # raw_input_dir = "images/"
    output_dir = "dataset_{}/".format(tiles_per_dim)
    files = os.listdir(raw_input_dir)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    # tiles_per_dim = 4

    for f in files:
        # TODO: add another for loop to do the same also for augmented tiles, but make sure OoD is from a previous image, not from a previous augmentation
        folder_output_dir = output_dir+f.split('.')[0]+'/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        os.mkdir(folder_output_dir)

        # add OOD from previous crops if exist:
        if len(crops_previous) > 0:
            idx = np.arange(len(crops_previous))
            np.random.shuffle(idx)
            for c_idx in idx[:tiles_per_dim]:
                cv2.imwrite(folder_output_dir+names_previous[c_idx], crops_previous[c_idx])
        im = cv2.imread(raw_input_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height // tiles_per_dim
        frac_w = width // tiles_per_dim
        i = 0
        crops_previous = []
        names_previous = []
        for h in range(tiles_per_dim):
            for w in range(tiles_per_dim):
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                cv2.imwrite(folder_output_dir + f[:-4] + "_{}.jpg".format(str(i).zfill(2)), crop)
                crops_previous.append(crop)
                names_previous.append(f[:-4] + "_{}_{}.jpg".format(str(i).zfill(2), str(-1))) # -1 indicates it will OoD for next image
                i = i + 1


def split_into_train_val_test(dataset_folder, portion_train, portion_val):
    train_test_val_dict = {}
    folders = os.listdir(dataset_folder)
    np.random.shuffle(folders)
    num_folders = len(folders)
    stop_idx_train = int(num_folders*portion_train)
    stop_idx_val = stop_idx_train+int(num_folders*portion_val)
    train_folders = folders[:stop_idx_train]
    val_folders = folders[stop_idx_train : stop_idx_val]
    test_folders = folders[stop_idx_val :]
    train_test_val_dict['train'] = train_folders
    train_test_val_dict['val'] = val_folders
    train_test_val_dict['test'] = test_folders
    save_obj(train_test_val_dict, 'train_test_val_dict')


def save_obj(obj, name, directory=''):
    with open(directory + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def load_obj(name, directory=''):
    with open(directory + name + '.pkl', 'rb') as f:
        return pkl.load(f)


def resize_image(image, max_size=None, resize_factor=None):
    import cv2
    import math
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
    return im_resized


if __name__ == '__main__':

    # TODO: need this for multiple tile sizes, as well as for documents
    # shredder("images/", 2)
    # shredder("images/", 4)
    # shredder("images/", 5)
    # shredder("documents/", 2)
    # shredder("documents/", 4)
    # shredder("documents/", 5)

    # split_into_train_val_test('dataset_2', 0.75, 0.15)
    d = load_obj('train_test_val_dict')
    print(len(d['train']))


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
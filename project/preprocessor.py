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
import keras


def shredder(raw_input_dir, tiles_per_dim):
    import cv2
    import os
    import numpy as np

    Xa = []
    Xb = []
    y = []

    crops_previous = []
    names_previous = []

    # TODO: need this for multiple tile sizes, as well as for documents

    # raw_input_dir = "images/"
    output_dir = "dataset_{}/".format(tiles_per_dim)
    files = os.listdir(raw_input_dir)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    # tiles_per_dim = 4

    for f in files:
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




shredder("images/", 2)
# shredder("images/", 4)
# shredder("images/", 5)
shredder("documents/", 2)
# shredder("documents/", 4)
# shredder("documents/", 5)


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
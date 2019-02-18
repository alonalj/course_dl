'''
# Data prep
dataset augmentation:
(1) crop images (into smaller crops only to not include hints (e.g. black edges) of which tile this is
(2) all other relevant augmentations we saw in the code provided for transfer learning in ex. 3
for image,
    for augmentation of image:
        shred in preprocessor into subfolders with labels in name ( check keras input?)
for each folder randomly select t ood tiles from another folder and add them
split folders into train, val, test parent-folderas randomly
'''

import os
import pickle as pkl
import numpy as np
from conf import Conf
import random
import cv2


def get_shapes(images):
    shape_to_count_dict = {}
    shape_majority_count = 0
    for im in images:
        current_shape = im.shape
        if current_shape in shape_to_count_dict.keys():
            shape_to_count_dict[current_shape] += 1
        else:
            shape_to_count_dict[current_shape] = 1
    for k in shape_to_count_dict.keys():
        if shape_majority_count < shape_to_count_dict[k]:
            shape_majority = k
            shape_majority_count = shape_to_count_dict[k]
    # print(shape_to_count_dict, shape_majority)
    return shape_to_count_dict, shape_majority


def reshape_ood_images_to_majority_shape(images, shape_majority):
    for i in range(len(images)):
        im = images[i]
        if im.shape != shape_majority:
            im = cv2.resize(im, (shape_majority[1], shape_majority[0]))
            images[i] = im
    return images


def add_similarity_channel(processed_images, original_images, c, n_channels=None, only_sim=False, sim_on_side=False):
    from scipy import spatial  # TODO: add to dependencies

    indices_identified_as_ood = []
    # print("GG", np.array(processed_images).shape)
    final_images = []
    ood_possible_by_shape = False
    img_shape_to_count_dict, shape_majority = get_shapes(original_images)
    mean_similarity_to_neighbors = []
    all_grayscale_tones = []
    original_images = reshape_ood_images_to_majority_shape(original_images, shape_majority)
    processed_images = reshape_ood_images_to_majority_shape(processed_images, shape_majority)

    for i in range(len(processed_images)):
        if only_sim:
            sim_layer = np.zeros((25, 25))
        elif sim_on_side:
            sim_layer = np.zeros((c.max_size, 4))
        else:
            sim_layer = np.zeros((c.max_size, c.max_size))
        im = original_images[i]
        mean_grayscale_value = np.mean(im)
        sum_similarity_to_neighbors = 0

        right_edge_original = im[:, -1].flatten()
        left_edge_original = im[:, 0].flatten()
        top_edge_original = im[0, :].flatten()
        bottom_edge_original = im[-1, :].flatten()
        row = 0
        # for m in [right_edge_original, left_edge_original, top_edge_original, bottom_edge_original]:
        #     print(len(m))
        for j in range(len(processed_images)):
            if i == j:
                continue
            potential_neighbor = original_images[j]
            # if potential_neighbor.shape != shape_majority:
            #     sim_layer[row, 0:4] = 0,0,0,0

            # potential_neighbor = cv2.resize(potential_neighbor, (224, 224))
            right_edge = potential_neighbor[:,-1].flatten()
            left_edge = potential_neighbor[:,0].flatten()
            top_edge = potential_neighbor[0, :].flatten()
            bottom_edge = potential_neighbor[-1, :].flatten()
            # for m in [right_edge, left_edge, top_edge, bottom_edge]:
            #     print(len(m))
            cosine_sim_rl = 1-spatial.distance.cosine(np.add(right_edge_original, 0.0001), np.add(left_edge, 0.0001))
            cosine_sim_lr = 1-spatial.distance.cosine(np.add(left_edge_original, 0.0001), np.add(right_edge, 0.0001))
            cosine_sim_tb = 1-spatial.distance.cosine(np.add(top_edge_original, 0.0001), np.add(bottom_edge, 0.0001))
            cosine_sim_bt = 1-spatial.distance.cosine(np.add(bottom_edge_original, 0.0001), np.add(top_edge, 0.0001))
            sim_layer[row,0:4] = cosine_sim_lr, cosine_sim_rl, cosine_sim_tb, cosine_sim_bt
            sum_similarity_to_neighbors += np.max(sim_layer[row, 0:4])
            row += 1
            # print(i, j, cosine_sim_lr)
        if only_sim:
            sim_layer = 255 * sim_layer
            final_image = sim_layer
        else:
            if sim_on_side:
                img = cv2.resize(processed_images[i], (c.max_size-4, c.max_size))
                if np.max(img) > 1:
                    sim_layer = 255 * sim_layer
                img = np.concatenate([img,sim_layer], 1)
                final_image = img
            else:
                final_image = np.zeros((c.max_size, c.max_size, 3))
                final_image[:,:,0] = cv2.resize(processed_images[i], (c.max_size,c.max_size))
                if np.max(final_image) > 1:
                    sim_layer = 255 * sim_layer
                final_image[:,:,1] = sim_layer

        final_images.append(final_image)  # each has two channels, the second channel has the similarities
        sum_similarity_to_neighbors = sum_similarity_to_neighbors / float(len(processed_images))
        mean_similarity_to_neighbors.append(sum_similarity_to_neighbors)
        all_grayscale_tones.append(mean_grayscale_value)

    # zipped = zip(final_images, all_grayscale_tones)
    # zipped = sorted(zipped, key=lambda x: x[1], reverse=True)  # sorting by grayscale tone
    # zipped = [item[0] for item in zipped]

    return final_images


def _shredder(raw_input_dir, c, output_dir):
    import os

    Xa = []
    Xb = []
    y = []

    add_t_OoDs = False

    # raw_input_dir = "images/"

    files = os.listdir(raw_input_dir)
    print("shredding for dictionary {}".format(c.data_split_dict))
    files_dict = load_obj(c.data_split_dict)
    files = files_dict
    # files = [f for f in files if "n01440764_7267" in f]
    list_of_folders = []
    # update this number for 4X4 crop 2X2 or 5X5 crops.
    # tiles_per_dim = 4
    # if data_type == 'train':
    #     if 'image' in raw_input_dir:
    #         print("shredding with crops for images")
    #         crop_start_w = range(0, 46, 15)
    #         crop_start_h = range(0, 46, 15)
    #     else:
    #         print("shredding with crops for docs")
    #         crop_start_w = range(0, 91, 30)
    #         crop_start_h = range(0, 91, 30)
    #     reshape_options = [False]
    # else:
    crop_start_w = [0]
    crop_start_h = [0]
    reshape_options = [False]


    crops_previous = []
    names_previous = []

    for f in files:

        filename = f.split('.')[0]
        folder_name = output_dir + filename
        folder_output_dir = folder_name + '/'
        list_of_folders.append(filename)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(folder_output_dir):
            os.mkdir(folder_output_dir)

        if add_t_OoDs:
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
                        # if random.random() < 0.5:  # sometimes place at the top, sometimes at the bottom
                        #     name = names_previous[c_idx]
                        #     name_split = name.split('_')
                        #     name_split[1] = '0'  # reducing image id to 0 so when sorted will appear first
                        #     name = name_split[0]+'_' + name_split[1] + '_' + name_split[2]+'_'+name_split[3]
                        #     cv2.imwrite(folder_output_dir + name, crops_previous[c_idx])
                        # else:
                        cv2.imwrite(folder_output_dir+names_previous[c_idx], crops_previous[c_idx])
            else:
                folder_output_dir_first_img = folder_output_dir  # first img has no "previous", will use last img later

        im = cv2.imread(raw_input_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        original_height, original_width = im.shape[0], im.shape[1]
        # augmentation - crop image  # TODO: remove this for ducments - probably easier due to white gap naturally present on page
            # cv2.imshow("cropped", cropped)

        height = im.shape[0]
        width = im.shape[1]
        # if original_height != height and original_width != width and reshape:
        #     try:
        #         im = cv2.resize(im,(original_height, original_width))
        #     except:
        #         continue

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

        if add_t_OoDs:
            # take OoD from last image in files as "previous" for first img
            if f == files[-1]:
                idx = np.arange(len(crops_previous))
                np.random.shuffle(idx)
                for c_idx in idx[:c.tiles_per_dim]:
                    cv2.imwrite(folder_output_dir_first_img + names_previous[c_idx], crops_previous[c_idx])

    # files_dict[data_type] = list_of_folders
    # print(files_dict)
    # save_obj(files_dict, c.data_split_dict)

#
# def split_into_train_val_test(dataset_folder, portion_train, portion_val, dict_name):
#     # call the main split for all files
#     train_test_val_dict = {}
#     folders_original = os.listdir(dataset_folder)
#     folders = list(set([folders_original[i].split('_cr')[0] for i in range(len(folders_original))]))
#     folders = [f for f in folders if 'DS_Store' not in f]
#     np.random.shuffle(folders)
#     num_folders = len(folders)
#     stop_idx_train = int(num_folders*portion_train)
#     stop_idx_val = stop_idx_train+int(num_folders*portion_val)
#     train_folders = folders[:stop_idx_train]
#     val_folders = folders[stop_idx_train : stop_idx_val]
#     test_folders = folders[stop_idx_val :]
#
#     train_folders = [f for f in folders_original if f.split('_cr')[0] in train_folders]
#     val_folders = [f for f in folders_original if f.split('_cr')[0] in val_folders]
#     test_folders = [f for f in folders_original if f.split('_cr')[0] in test_folders]
#
#     train_test_val_dict['train'] = train_folders
#     train_test_val_dict['val'] = val_folders
#     train_test_val_dict['test'] = test_folders
#     save_obj(train_test_val_dict, dict_name)


# def split_into_train_val_test(dataset_folder, n_test, dict_name):
#     train_test_val_dict = {}
#     folders_original = os.listdir(dataset_folder)
#     folders = list(set([folders_original[i].split('_cr')[0] for i in range(len(folders_original))]))
#     folders = [f for f in folders if 'DS_Store' not in f]
#     np.random.shuffle(folders)
#     num_folders = len(folders)
#     train_folders = folders[:-60]
#     val_folders = folders[-60 : -30]
#     test_folders = folders[-30 :]
#
#     train_folders = [f for f in folders_original if f.split('_cr')[0] in train_folders]
#     val_folders = [f for f in folders_original if f.split('_cr')[0] in val_folders]
#     test_folders = [f for f in folders_original if f.split('_cr')[0] in test_folders]
#
#     train_test_val_dict['train'] = train_folders
#     train_test_val_dict['val'] = val_folders
#     train_test_val_dict['test'] = test_folders
#     save_obj(train_test_val_dict, dict_name)


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
    if 'False' in c.data_split_dict:
        output_dir = "dataset_{}_isImg_False/".format(c.tiles_per_dim)
    else:
        output_dir = "dataset_{}_isImg_True/".format(c.tiles_per_dim)
    if os.path.exists(output_dir):
        print("Already shredded for: isImg {} and n_tiles_per_dim {}".format(c.is_images, c.tiles_per_dim))
        return
    dict_name = c.data_split_dict
    # if 'doc' in c.data_split_dict:
    #     split_into_train_val_test('documents', 30, dict_name)
    # else:
    #     split_into_train_val_test('images', 30, dict_name)
    split_train_val_test('True' in c.data_split_dict,0.8,0.1)
    # else:
    d = load_obj(dict_name)
    # TODO: need this for multiple tile sizes, as well as for documents

    if 'False' in c.data_split_dict:
        _shredder("documents/", c, output_dir)
    else:
        _shredder("images/", c, output_dir)


def shred_with_similarity_channel(isImg, tiles_per_dim, c, OUTPUT_DIR):
    Xa = []
    Xb = []
    y = []

    if isImg:
        IM_DIR = "images/"
    else:
        IM_DIR = "documents/"

    # OUTPUT_DIR = "dataset_rows_cols_{}_isImg_{}/".format(tiles_per_dim, isImg)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    else:
        print("Already shredded")
        return

    files = os.listdir(IM_DIR)
    for f in files:
        all_crops = []
        im = cv2.imread(IM_DIR+f)
        im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height//tiles_per_dim
        frac_w = width//tiles_per_dim
        i=0
        crop_names = []
        for h in range(tiles_per_dim):
            for w in range(tiles_per_dim):

                crop = im[h*frac_h:(h+1)*frac_h,w*frac_w:(w+1)*frac_w]
                all_crops.append(crop)
                crop_names.append(OUTPUT_DIR + f[:-4] + "_{}.jpg".format(str(i).zfill(2)))
                i+=1

        # print("before", crop_names)
        zipped = zip(crop_names,all_crops)
        result = sorted(zipped,key=lambda x: x[0])
        crop_names, all_crops = [item[0] for item in result], [item[1] for item in result]
        # print("after", crop_names)
        # print("before", all_crops)
        # random.shuffle(all_crops)
        # print("after", all_crops)
        all_crops = add_similarity_channel(all_crops, all_crops, c, only_sim=False, sim_on_side=True)
        i = 0
        for crop in all_crops:
            # crop = np.resize(crop, (c.max_size, c.max_size)) # TODO; remove if using similarity channel
            cv2.imwrite(OUTPUT_DIR + f[:-4] + "_{}.jpg".format(str(i).zfill(2)), crop)
            i+=1

def shred_for_ood_pairs(isImg):
    Xa = []
    Xb = []
    y = []

    if isImg:
        IM_DIR = "images/"
    else:
        IM_DIR = "documents/"

    OUTPUT_DIR = "dataset_for_ood_pairs_isImg_{}/".format(IM_DIR == "images/")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    else:
        print("Already shredded")
        return

    files = os.listdir(IM_DIR)
    for f in files:
        for tiles_per_dim in [2,4,5]:
            all_crops = []
            im = cv2.imread(IM_DIR+f)
            im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            height = im.shape[0]
            width = im.shape[1]
            frac_h = height//tiles_per_dim
            frac_w = width//tiles_per_dim
            i=0
            for h in range(tiles_per_dim):
                for w in range(tiles_per_dim):

                    crop = im[h*frac_h:(h+1)*frac_h,w*frac_w:(w+1)*frac_w]
                    all_crops.append(crop)

            # all_crops = add_similarity_channel(all_crops, all_crops, c, n_channels=3)
            i = 0
            for crop in all_crops:
                cv2.imwrite(OUTPUT_DIR + f[:-4] + "_{}_t_{}.jpg".format(str(i).zfill(2), tiles_per_dim), crop)
                i+=1


def get_relevant_files(data_type, c):
    path = c.output_dir
    relevant_files = load_obj('files_{}_img_{}'.format(data_type, c.is_images))
    folders = os.listdir(path)
    if c.is_images:
        relevant_files = [f for f in folders if f.split('.')[0] + '.JPEG' in relevant_files]  # tTODO: for images too
    else:
        relevant_files = [f for f in folders if f.split('.')[0][:-3] + '.jpg' in relevant_files]  # tTODO: for images too
    return relevant_files


def create_normalization_stats(c, rows_or_cols):
    import glob
    if os.path.exists('mean_{}_isImg_{}.pkl'.format(c.tiles_per_dim, c.is_images)) and os.path.exists('std_{}_isImg_{}.pkl'.format(c.tiles_per_dim, c.is_images)):
        print("already calculated mean, std stats")
        return
    sum_imgs = np.zeros((c.max_size, c.max_size))
    subt_imgs_2 = np.zeros((c.max_size, c.max_size))
    count_images = 0
    train_files = get_relevant_files("train", c)
    for file in train_files:
        im = cv2.imread(c.output_dir+file)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.resize(im, (c.max_size, c.max_size))
        sum_imgs += im
        count_images += 1
    mean_image = sum_imgs / float(count_images)
    for file in train_files:
        im = cv2.imread(c.output_dir+file)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.resize(im, (c.max_size, c.max_size))
        subt_imgs_2 += (im - mean_image)**2
    stdev_image = np.sqrt(subt_imgs_2 / float(count_images-1))

    save_obj(mean_image, 'mean_{}_isImg_{}'.format(c.tiles_per_dim, c.is_images))
    save_obj(stdev_image, 'std_{}_isImg_{}'.format(c.tiles_per_dim, c.is_images))


def preprocess_image(im, c):

    MEAN = load_obj('mean_{}_isImg_{}'.format(c.tiles_per_dim, c.is_images))
    STD = load_obj('std_{}_isImg_{}'.format(c.tiles_per_dim, c.is_images))

    def normalize(im, is_img=True):
        im = (im - MEAN) / STD
        return im

    try:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    except:
        pass
    im = cv2.resize(im, (c.max_size, c.max_size))
    im = normalize(im, c.is_images)
    im = np.expand_dims(im, -1)
    return im

def shred_for_img_vs_doc():
    Xa = []
    Xb = []
    y = []

    for filedir in ['images/', 'documents/']:
        if filedir == 'images/':
            sub_folder = '1/'
        else:
            sub_folder = '0/'
        IM_DIR = filedir
        OUTPUT_DIR_TRAIN= "img_vs_doc/"
        OUTPUT_DIR_TEST = "img_vs_doc_val/"
        # if not os.path.exists(OUTPUT_DIR_TEST):
        #     os.mkdir(OUTPUT_DIR_TEST)
        # if not os.path.exists(OUTPUT_DIR_TRAIN):
        #     os.mkdir(OUTPUT_DIR_TEST)
        # if not os.path.exists(OUTPUT_DIR_TRAIN+sub_folder):
        #     os.mkdir(OUTPUT_DIR_TRAIN+sub_folder)
        # if not os.path.exists(OUTPUT_DIR_TEST+sub_folder):
        #     os.mkdir(OUTPUT_DIR_TEST+sub_folder)
        files = np.array(os.listdir(IM_DIR))
        np.random.shuffle(files)
        files_train = files[:int(len(files)*0.8)]
        files_test = files[int(len(files) * 0.8):]
        # update this number for 4X4 crop 2X2 or 5X5 crops.
        # tiles_per_dim = 2
        for tiles_per_dim in [2,4,5]:
            for f in files:
                if f in files_train:
                    OUTPUT_DIR = OUTPUT_DIR_TRAIN
                else:
                    OUTPUT_DIR = OUTPUT_DIR_TEST
                im = cv2.imread(IM_DIR + f)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                height = im.shape[0]
                width = im.shape[1]
                frac_h = height // tiles_per_dim
                frac_w = width // tiles_per_dim
                i = 0
                for h in range(tiles_per_dim):
                    for w in range(tiles_per_dim):
                        crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                        crop = np.resize(crop, (32,32))
                        cv2.imwrite(OUTPUT_DIR + sub_folder + f[:-4] + "_{}_{}.jpg".format(str(i).zfill(2), str(tiles_per_dim)), crop)
                        i = i + 1


def get_row_col_label(f, c, is_rows):
    t = c.tiles_per_dim
    if c.is_images:
        label = f.split('.')[1].split('_')[1]
    else:
        label = f.split('.')[0][-2:]
    label = int(label)
    if is_rows:
        label = int(label/t)
    else:
        label = int(label % t)
    return label


def shred_original_ta():
    import cv2
    import os
    import numpy as np

    Xa = []
    Xb = []
    y = []

    IM_DIR = "images/"
    OUTPUT_DIR = "output/"
    files = os.listdir(IM_DIR)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 4

    for f in files:
        im = cv2.imread(IM_DIR + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height // tiles_per_dim
        frac_w = width // tiles_per_dim
        i = 0
        for h in range(tiles_per_dim):
            for w in range(tiles_per_dim):
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                cv2.imwrite(OUTPUT_DIR + f[:-4] + "_{}.jpg".format(str(i).zfill(2)), crop)
                i = i + 1


def shredder_original(isImg, tiles_per_dim, c, OUTPUT_DIR, dict=[]):

    Xa = []
    Xb = []
    y = []

    if isImg:
        IM_DIR = "images/"
    else:
        IM_DIR = "documents/"

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    else:
        print("Already shredded")
        return

    if len(dict) > 1:
        files_dict = load_obj(c.data_split_dict)
        files = files_dict
    else:
        files = os.listdir(IM_DIR)

    for f in files:
        im = cv2.imread(IM_DIR+f)
        im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        height = im.shape[0]
        width = im.shape[1]
        frac_h = height//tiles_per_dim
        frac_w = width//tiles_per_dim
        i=0
        for h in range(tiles_per_dim):
            for w in range(tiles_per_dim):

                crop = im[h*frac_h:(h+1)*frac_h,w*frac_w:(w+1)*frac_w]
                cv2.imwrite(OUTPUT_DIR+f[:-4]+"_{}.jpg".format(str(i).zfill(2)),crop)
                i=i+1


def split_train_val_test(is_img, ratio_train=0.8, ratio_val=0.1, ratio_test=0.1):
    try:
        files_train = load_obj("files_train_img_{}".format(is_img))
        files_val = load_obj("files_val_img_{}".format(is_img))
        files_test = load_obj("files_test_img_{}".format(is_img))
        print("already split files into train val test, loading..")
    except:
        print("splitting into train val test")
        if is_img:
            files = os.listdir('images/')
        else:
            files = os.listdir('documents/')
        files = np.array(files)
        np.random.shuffle(files)
        files_train = files[:int(len(files) * ratio_train)]
        files_val = files[int(len(files) * ratio_train):int(len(files) * ratio_train)+int(len(files) * ratio_val)]
        files_test = [f for f in files if (f not in files_train) and (f not in files_val)]
        save_obj(files_train, "files_train_img_{}".format(is_img))
        save_obj(files_val, "files_val_img_{}".format(is_img))
        save_obj(files_val, "files_test_img_{}".format(is_img))
        assert len([f for f in files_test if f in files_train]) == 0
        assert len([f for f in files_test if f in files_val]) == 0
        assert len([f for f in files_val if f in files_train]) == 0
    return files_train, files_val, files_test


def create_rows_cols_folders_by_class(tiles_per_dim, isImg, rows_or_cols):
    import shutil

    Xa = []
    Xb = []
    y = []


    OUTPUT_DIR_TRAIN = "{}_{}_isImg_{}/".format(rows_or_cols, tiles_per_dim, isImg)
    OUTPUT_DIR_VAL = "{}_{}_isImg_{}_val/".format(rows_or_cols, tiles_per_dim, isImg)
    OUTPUT_DIR_TEST = "{}_{}_isImg_{}_test/".format(rows_or_cols, tiles_per_dim, isImg)

    if not os.path.exists(OUTPUT_DIR_TRAIN):
        os.mkdir(OUTPUT_DIR_TRAIN)
        os.mkdir(OUTPUT_DIR_VAL)
        os.mkdir(OUTPUT_DIR_TEST)
    else:
        print("folders already created.")
        return

    files_train, files_val, files_test = split_train_val_test(isImg)

    # for rows_or_cols in ["rows", "cols"]:
    IM_DIR = "dataset_rows_cols_{}_isImg_{}/".format(tiles_per_dim, isImg)
    files = np.array(os.listdir(IM_DIR))

    for f in os.listdir(IM_DIR):
        label = int(f.split('_')[-1].split('.')[0])
        if isImg:
            pre_shredder_file_name = f.split('.')[0]
            pre_shredder_file_name = pre_shredder_file_name + '.JPEG'
        else:
            pre_shredder_file_name = f.split('_')[0] + '_' + f.split('_')[1]
            pre_shredder_file_name = pre_shredder_file_name + '.jpg'

        label = get_row_col_label(label, tiles_per_dim, rows_or_cols == "rows")
        label = str(label)
        if not os.path.exists(OUTPUT_DIR_TRAIN+label):
            os.mkdir(OUTPUT_DIR_TRAIN+label)
            os.mkdir(OUTPUT_DIR_VAL+label)
            os.mkdir(OUTPUT_DIR_TEST + label)
        if pre_shredder_file_name in files_train:
            OUTPUT_DIR = OUTPUT_DIR_TRAIN
        elif pre_shredder_file_name in files_val:
            OUTPUT_DIR = OUTPUT_DIR_VAL
        else:
            OUTPUT_DIR = OUTPUT_DIR_TEST

        shutil.copy(IM_DIR+f,OUTPUT_DIR+label+"/")


def create_ood_non_ood_pairs(c):

    '''
    Creates two folders (two classes):
    0 - contains pairs that are from the same image
    1 - contains pairs that are not from the same image
    (The model will take as input two images, and will produce either 0 or 1)
    At prediction time, we will take the majority vote over the paris
    (e.g. if for image i all pairs (i,j) where j != i produced 1, then i is OoD)
    '''
    import shutil
    import glob

    Xa = []
    Xb = []
    y = []

    isImg = c.is_images
    base_name = "ood_isImg_{}".format(isImg)
    OUTPUT_DIR_TRAIN = base_name + "/"
    OUTPUT_DIR_VAL = base_name + "_val/"
    OUTPUT_DIR_TEST = base_name + "_test/"

    if not os.path.exists(OUTPUT_DIR_TRAIN):
        os.mkdir(OUTPUT_DIR_TRAIN)
        os.mkdir(OUTPUT_DIR_VAL)
        os.mkdir(OUTPUT_DIR_TEST)
    else:
        print("folders already created.")
        return

    files_train, files_val, files_test = split_train_val_test(isImg)

    # for rows_or_cols in ["rows", "cols"]:
    IM_DIR = "dataset_for_ood_pairs_isImg_{}/".format(isImg)

    folder_counter = 0
    for label in [0,1]:
        label = str(label)
        if not os.path.exists(OUTPUT_DIR_TRAIN + label):
            os.mkdir(OUTPUT_DIR_TRAIN + label)
            os.mkdir(OUTPUT_DIR_VAL + label)
            os.mkdir(OUTPUT_DIR_TEST + label)
    for tiles_per_dim in [2,4,5]:
        print(tiles_per_dim)
        for dataset in ["train", "val", "test"]:
            if dataset == "train":
                dataset_files = files_train
                OUTPUT_DIR = OUTPUT_DIR_TRAIN
            elif dataset == "val":
                dataset_files = files_val
                OUTPUT_DIR = OUTPUT_DIR_VAL
            else:
                dataset_files = files_test
                OUTPUT_DIR = OUTPUT_DIR_TEST

            files_for_t = glob.glob(IM_DIR + '*t_{}*'.format(tiles_per_dim))
            np.random.shuffle(files_for_t)
            files_for_t = files_for_t[:4000]
            files_for_t = [f.split('/')[-1] for f in files_for_t]

            if isImg:
                files_for_t = [f for f in files_for_t if f.split('.')[0] + '.JPEG' in dataset_files]
                image_ids = set([f.split('_')[1] for f in files_for_t])
            else:
                files_for_t = [f for f in files_for_t if f.split('_')[0]+"_"+f.split('_')[1] + '.jpg' in dataset_files]
                image_ids = set([f.split('_')[0]+'_'+f.split('_')[1] for f in files_for_t])

            for im_id in image_ids:
                tiles_in_distribution = []
                tiles_ood = []
                # get files in distribution and out of distribution given an image id (in = all tiles belonging to img)
                for f in files_for_t:
                    if isImg:
                        f_im_id = f.split('_')[1]
                    else:
                        f_im_id = f.split('_')[0]+'_'+f.split('_')[1]
                    if im_id == f_im_id:  # tiles are from same image
                        tiles_in_distribution.append(f)
                    else:
                        tiles_ood.append(f)

                # create pairs in distribution and ood with even number in each class
                # 1. in distribution:
                count_pairs_per_class = 0
                label = str(0)
                for i in range(len(tiles_in_distribution)):
                    for j in range(i+1,len(tiles_in_distribution)):
                        f1 = tiles_in_distribution[i]
                        f2 = tiles_in_distribution[j]
                        combined_images = []
                        # print(f1, f2)
                        # for f in [f1, f2]:
                        #     im = cv2.imread(IM_DIR+f)
                        #     im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                        #     im = cv2.resize(im, (c.max_size, c.max_size))
                        #     combined_images.append(im)
                        # combined_images = np.concatenate(combined_images, axis=1)
                        # cv2.imwrite(OUTPUT_DIR + label + "/"+str(folder_counter)+'.jpg', combined_images)
                        os.makedirs(OUTPUT_DIR +label+ '/'+ str(folder_counter))
                        shutil.copy(IM_DIR + f1, OUTPUT_DIR + label + "/"+ str(folder_counter) +"/")
                        shutil.copy(IM_DIR + f2, OUTPUT_DIR + label + "/"+ str(folder_counter) +"/")
                        count_pairs_per_class += 1
                        folder_counter += 1
                # 2. ood:
                label = str(1)
                for i in range(count_pairs_per_class):
                    f1 = random.choice(tiles_ood)
                    f2 = random.choice(tiles_in_distribution)
                    combined_images = []
                    # print("o", f1, f2)
                    # for f in [f1, f2]:
                    #     im = cv2.imread(IM_DIR+f)
                    #     im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                    #     im = cv2.resize(im, (c.max_size, c.max_size))
                    #     combined_images.append(im)
                    # combined_images = np.concatenate(combined_images, axis=1)
                    # cv2.imwrite(OUTPUT_DIR + label + "/" + str(folder_counter) + '.jpg', combined_images)
                    os.makedirs(OUTPUT_DIR + label + '/' + str(folder_counter))
                    shutil.copy(IM_DIR + f1, OUTPUT_DIR + label + "/" + str(folder_counter) + "/")
                    shutil.copy(IM_DIR + f2, OUTPUT_DIR + label + "/" + str(folder_counter) + "/")
                    folder_counter += 1

# def create_ood_non_ood_pairs_OLD(c):
#
#     '''
#     Creates two folders (two classes):
#     0 - contains pairs that are from the same image
#     1 - contains pairs that are not from the same image
#     (The model will take as input two images, and will produce either 0 or 1)
#     At prediction time, we will take the majority vote over the paris
#     (e.g. if for image i all pairs (i,j) where j != i produced 1, then i is OoD)
#     '''
#     import shutil
#     import glob
#
#     Xa = []
#     Xb = []
#     y = []
#
#     isImg = c.is_images
#     base_name = "ood_isImg_{}".format(isImg)
#     OUTPUT_DIR_TRAIN = base_name + "/"
#     OUTPUT_DIR_VAL = base_name + "_val/"
#     OUTPUT_DIR_TEST = base_name + "_test/"
#
#     if not os.path.exists(OUTPUT_DIR_TRAIN):
#         os.mkdir(OUTPUT_DIR_TRAIN)
#         os.mkdir(OUTPUT_DIR_VAL)
#         os.mkdir(OUTPUT_DIR_TEST)
#     else:
#         print("folders already created.")
#         return
#
#     files_train, files_val, files_test = split_train_val_test(isImg)
#
#     # for rows_or_cols in ["rows", "cols"]:
#     IM_DIR = "dataset_for_ood_pairs_isImg_{}/".format(isImg)
#
#     folder_counter = 0
#     for label in [0,1]:
#         label = str(label)
#         if not os.path.exists(OUTPUT_DIR_TRAIN + label):
#             os.mkdir(OUTPUT_DIR_TRAIN + label)
#             os.mkdir(OUTPUT_DIR_VAL + label)
#             os.mkdir(OUTPUT_DIR_TEST + label)
#     for tiles_per_dim in [2,4,5]:
#         print(tiles_per_dim)
#         for dataset in ["train", "val", "test"]:
#             if dataset == "train":
#                 dataset_files = files_train
#                 OUTPUT_DIR = OUTPUT_DIR_TRAIN
#             elif dataset == "val":
#                 dataset_files = files_val
#                 OUTPUT_DIR = OUTPUT_DIR_VAL
#             else:
#                 dataset_files = files_test
#                 OUTPUT_DIR = OUTPUT_DIR_TEST
#
#             files_for_t = glob.glob(IM_DIR + '*t_{}*'.format(tiles_per_dim))
#             np.random.shuffle(files_for_t)
#             files_for_t = files_for_t[:4000]
#             files_for_t = [f.split('/')[-1] for f in files_for_t]
#
#             if isImg:
#                 files_for_t = [f for f in files_for_t if f.split('.')[0] + '.JPEG' in dataset_files]
#                 image_ids = set([f.split('_')[1] for f in files_for_t])
#             else:
#                 files_for_t = [f for f in files_for_t if f.split('_')[0]+"_"+f.split('_')[1] + '.jpg' in dataset_files]
#                 image_ids = set([f.split('_')[0]+'_'+f.split('_')[1] for f in files_for_t])
#
#             for im_id in image_ids:
#                 tiles_in_distribution = []
#                 tiles_ood = []
#                 # get files in distribution and out of distribution given an image id (in = all tiles belonging to img)
#                 for f in files_for_t:
#                     if isImg:
#                         f_im_id = f.split('_')[1]
#                     else:
#                         f_im_id = f.split('_')[0]+'_'+f.split('_')[1]
#                     if im_id == f_im_id:  # tiles are from same image
#                         tiles_in_distribution.append(f)
#                     else:
#                         tiles_ood.append(f)
#
#                 # create pairs in distribution and ood with even number in each class
#                 # 1. in distribution:
#                 count_pairs_per_class = 0
#                 label = str(0)
#                 for i in range(len(tiles_in_distribution)):
#                     for j in range(i+1,len(tiles_in_distribution)):
#                         f1 = tiles_in_distribution[i]
#                         f2 = tiles_in_distribution[j]
#                         combined_images = []
#                         # print(f1, f2)
#                         # for f in [f1, f2]:
#                         #     im = cv2.imread(IM_DIR+f)
#                         #     im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#                         #     im = cv2.resize(im, (c.max_size, c.max_size))
#                             # combined_images.append(im)
#                         # combined_images = np.concatenate(combined_images, axis=1)
#                         # cv2.imwrite(OUTPUT_DIR + label + "/"+str(folder_counter)+'.jpg', combined_images)
#                         os.makedirs(OUTPUT_DIR +label+ '/'+ str(folder_counter))
#                         shutil.copy(IM_DIR + f1, OUTPUT_DIR + label + "/"+ str(folder_counter) +"/")
#                         shutil.copy(IM_DIR + f2, OUTPUT_DIR + label + "/"+ str(folder_counter) +"/")
#                         count_pairs_per_class += 1
#                         folder_counter += 1
#                 # 2. ood:
#                 label = str(1)
#                 for i in range(count_pairs_per_class):
#                     f1 = random.choice(tiles_ood)
#                     f2 = random.choice(tiles_in_distribution)
#                     combined_images = []
#                     # print("o", f1, f2)
#                     # for f in [f1, f2]:
#                     #     im = cv2.imread(IM_DIR+f)
#                     #     im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#                     #     im = cv2.resize(im, (c.max_size, c.max_size))
#                         # combined_images.append(im)
#                     # combined_images = np.concatenate(combined_images, axis=1)
#                     # cv2.imwrite(OUTPUT_DIR + label + "/" + str(folder_counter) + '.jpg', combined_images)
#                     os.makedirs(OUTPUT_DIR + label + '/' + str(folder_counter))
#                     shutil.copy(IM_DIR + f1, OUTPUT_DIR + label + "/" + str(folder_counter) + "/")
#                     shutil.copy(IM_DIR + f2, OUTPUT_DIR + label + "/" + str(folder_counter) + "/")
#                     folder_counter += 1

# split_train_val_test(True)
# shred_for_ood_pairs(True)
# c = Conf()
# c.is_images = True
# c.tiles_per_dim = 2
# create_ood_non_ood_pairs(c)
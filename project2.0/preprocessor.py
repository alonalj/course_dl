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
from scipy import spatial


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
#
def calc_cosine_sim_on_edges(im1, im2, disregard_whites=False):

    right_edge_original = im1[:, -1].flatten()
    left_edge_original = im1[:, 0].flatten()
    top_edge_original = im1[0, :].flatten()
    bottom_edge_original = im1[-1, :].flatten()

    right_edge = im2[:, -1].flatten()
    left_edge = im2[:, 0].flatten()
    top_edge = im2[0, :].flatten()
    bottom_edge = im2[-1, :].flatten()
    # for m in [right_edge, left_edge, top_edge, bottom_edge]:
    #     print(len(m))
    cosine_sim_rl = 1 - spatial.distance.cosine(np.add(right_edge_original, 0.0001), np.add(left_edge, 0.0001))
    cosine_sim_lr = 1 - spatial.distance.cosine(np.add(left_edge_original, 0.0001), np.add(right_edge, 0.0001))
    cosine_sim_tb = 1 - spatial.distance.cosine(np.add(top_edge_original, 0.0001), np.add(bottom_edge, 0.0001))
    cosine_sim_bt = 1 - spatial.distance.cosine(np.add(bottom_edge_original, 0.0001), np.add(top_edge, 0.0001))

    if disregard_whites: # handles document edges with all whites, making max cosine 1
        tuples = [(cosine_sim_rl, right_edge_original), (cosine_sim_lr, left_edge_original),
                  (cosine_sim_tb, top_edge_original), (cosine_sim_bt, bottom_edge_original)]
        results = []
        for tix in range(len(tuples)):
            if tuples[tix][0] > 0.9 and len(set(tuples[tix][1])) == 1:
                results.append(0)
            else:
                results.append(tuples[tix][0])
        cosine_sim_rl, cosine_sim_lr, cosine_sim_tb, cosine_sim_bt = results
    return cosine_sim_lr, cosine_sim_rl, cosine_sim_tb, cosine_sim_bt


def calc_cosine_sim_on_single_edge(pos_1, pos_2,
                    im_ix_1, im_ix_2,
                    images
                ):
    is_horizontal_neighbors = np.abs(pos_1[0] - pos_2[0]) == 0 and np.abs(pos_1[1] - pos_2[1]) == 1
    is_vertical_neighbors = np.abs(pos_1[0] - pos_2[0]) == 1 and np.abs(pos_1[1] - pos_2[1]) == 0

    image_1 = images[im_ix_1]
    image_2 = images[im_ix_2]
    if image_1.shape != image_2.shape:  # TODO ! - put this at the bottom and make sure this would mean similarity = 0
        image_1 = cv2.resize(image_1, (image_2.shape[1],image_2.shape[0]))

    if is_horizontal_neighbors:
        if pos_1[1] == pos_2[1] - 1: # 1 is left to 2
            edge_1 = image_1[:, -1] # right most column of pixels
            edge_2 = image_2[:, 0]
        else:
            edge_1 = image_1[:, 0] # left most column pf pixels
            edge_2 = image_2[:, -1]
    elif is_vertical_neighbors:
        if pos_1[0] == pos_2[0] - 1: # 1 above 2
            edge_1 = image_1[-1,:]
            edge_2 = image_2[0,:]
        else:
            edge_1 = image_1[0,:]
            edge_2 = image_2[-1,:]
    else:
        print("Invalid neighbors")
        print(pos_1, pos_2)
        return

    edge_1, edge_2 = edge_1.flatten(), edge_2.flatten()
    edge_cosine_sim = 1 - spatial.distance.cosine(np.add(edge_1, 0.0001), np.add(edge_2, 0.0001))
    return edge_cosine_sim


def add_similarity_channel(processed_images, original_images, c, n_channels=None, only_sim=False, sim_on_side=False, sim_layer_size=25, random_shuffle=False):

    indices_identified_as_ood = []
    # print("GG", np.array(processed_images).shape)
    final_images = []
    ood_possible_by_shape = False
    img_shape_to_count_dict, shape_majority = get_shapes(original_images)
    mean_similarity_to_neighbors = []
    all_grayscale_tones = []
    original_images = reshape_ood_images_to_majority_shape(original_images, shape_majority)
    processed_images = reshape_ood_images_to_majority_shape(processed_images, shape_majority)

    if random_shuffle:
        zipped = list(zip(processed_images,original_images,range(len(original_images))))
        random.shuffle(zipped)
        processed_images, original_images, indices = zip(*zipped)

    for i in range(len(processed_images)):
        if only_sim:
            sim_layer = np.zeros((sim_layer_size, sim_layer_size))
        elif sim_on_side:
            sim_layer = np.zeros((c.max_size, 4))
        else:
            sim_layer = np.zeros((c.max_size, c.max_size))
        im = original_images[i]
        mean_grayscale_value = np.mean(im)
        sum_similarity_to_neighbors = 0


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
            cosine_sim_lr, cosine_sim_rl, cosine_sim_tb, cosine_sim_bt = calc_cosine_sim_on_edges(im, potential_neighbor)
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

    if random_shuffle:
        return final_images, indices

    return final_images


def shredder_with_oods(raw_input_dir, c, output_dir, add_t_OoDs=True):
    import os
    import shutil

    Xa = []
    Xb = []
    y = []

    # raw_input_dir = "images/"
    if os.path.exists(output_dir):
        print("already created folder")
        return
        # shutil.rmtree(output_dir)
    files = os.listdir(raw_input_dir)
    if c.data_split_dict:
        print("shredding for dictionary {}".format(c.data_split_dict))
        files_dict = load_obj(c.data_split_dict)
        files = files_dict
    list_of_folders = []

    crop_start_w = [0]
    crop_start_h = [0]
    reshape_options = [False]

    crops_previous = []
    names_previous = []

    for f in files:
        if 'DS' in f:
            continue

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
                n_oods = random.randint(0,c.tiles_per_dim)
                for c_idx in idx[:n_oods]:
                    if random.random() < 0:
                        null_img = np.zeros((c.max_size, c.max_size))
                        cv2.imwrite(folder_output_dir + names_previous[c_idx], null_img)
                    else:
                        cv2.imwrite(folder_output_dir+names_previous[c_idx], crops_previous[c_idx])
            else:
                folder_output_dir_first_img = folder_output_dir  # first img has no "previous", will use last img later

        im = cv2.imread(raw_input_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        original_height, original_width = im.shape[0], im.shape[1]

        height = im.shape[0]
        width = im.shape[1]

        frac_h = height // c.tiles_per_dim
        frac_w = width // c.tiles_per_dim
        i = 0
        crops_previous = []
        names_previous = []
        for h in range(c.tiles_per_dim):
            for w in range(c.tiles_per_dim):
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                cv2.imwrite(folder_output_dir + f[:-4] + "_{}_{}.jpg".format(str(i).zfill(2), str(0)), crop)
                crops_previous.append(crop)
                names_previous.append(f[:-4] + "_{}_{}.jpg".format(str(i).zfill(2), str(1))) # -1 indicates it will OoD for next image
                i = i + 1

        if add_t_OoDs:
            # take OoD from last image in files as "previous" for first img
            if f == files[-1]:
                idx = np.arange(len(crops_previous))
                np.random.shuffle(idx)
                for c_idx in idx[:c.tiles_per_dim]:
                    cv2.imwrite(folder_output_dir_first_img + names_previous[c_idx], crops_previous[c_idx])


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

        w = shape[1]
        h = shape[0]
        slack_w = max_size - w  # padding size w
        slack_h = max_size - h  # padding size h

        import random
        start_w = 0
        end_w = start_w + w
        start_h = 0
        end_h = start_h + h
        base_size = max_size, max_size
        base = np.zeros(base_size, dtype=np.uint8)
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
    split_train_val_test('True' in c.data_split_dict,0.8,0.1)
    d = load_obj(dict_name)

    if 'False' in c.data_split_dict:
        shredder_with_oods("documents/", c, output_dir)
    else:
        shredder_with_oods("images/", c, output_dir)


def shred_with_similarity_channel(isImg, tiles_per_dim, c, OUTPUT_DIR, add_sim=True):
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

    files = os.listdir(IM_DIR)
    for f in files:
        if 'DS' in f:
            continue
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

        zipped = zip(crop_names,all_crops)
        result = sorted(zipped,key=lambda x: x[0])
        crop_names, all_crops = [item[0] for item in result], [item[1] for item in result]

        if add_sim:
            all_crops = add_similarity_channel(all_crops, all_crops, c, only_sim=False, sim_on_side=True)
        i = 0
        for crop in all_crops:
            cv2.imwrite(OUTPUT_DIR + f[:-4] + "_{}.jpg".format(str(i).zfill(2)), crop)
            i+=1


def shred_for_ood_pairs(isImg, c):
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
        for tiles_per_dim in [2]: #TODO
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

            all_crops = add_similarity_channel(all_crops, all_crops, c, only_sim=True,sim_layer_size=32)
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

        files = np.array(os.listdir(IM_DIR))
        np.random.shuffle(files)
        files_train = files[:int(len(files)*0.8)]
        files_test = files[int(len(files) * 0.8):]

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


def make_train_val_test_dirs(base_name):
    OUTPUT_DIR_TRAIN = base_name + "/"
    OUTPUT_DIR_VAL = base_name + "_val/"
    OUTPUT_DIR_TEST = base_name + "_test/"

    if os.path.exists(OUTPUT_DIR_TRAIN):
        print("folders already created.")
        return []
    else:
        os.mkdir(OUTPUT_DIR_TRAIN)
        os.mkdir(OUTPUT_DIR_VAL)
        os.mkdir(OUTPUT_DIR_TEST)

    return [OUTPUT_DIR_TRAIN, OUTPUT_DIR_VAL, OUTPUT_DIR_TEST]


def create_ood_files_by_similarity(input_path,output_path, c):
    import glob
    files_split = split_train_val_test(c.is_images)
    output_path_split = make_train_val_test_dirs(output_path)
    if len(output_path_split) < 1:
        return
    folders = glob.glob(input_path + '/*')
    short_folder_names = [f.split('/')[-1] for f in glob.glob(input_path + '/*')]

    for j in range(3):
        files_for_data_type = files_split[j]
        output_path = output_path_split[j]
        if c.is_images:
            folders_for_data_type = [f for f in folders if f.split('/')[-1]+'.JPEG' in files_for_data_type]
        else:
            folders_for_data_type = [f for f in folders if f.split('/')[-1]+'.jpg' in files_for_data_type]
        for i in ['/0','/1']:
            if not os.path.exists(output_path+i):
                os.makedirs(output_path+i)
        for folder in folders_for_data_type:
            files_in_folder = glob.glob(folder+'/*')
            images = []
            labels = []
            file_names = []
            for f in files_in_folder:
                file_name = f.split('/')[-1]
                file_names.append(file_name)
                im = cv2.imread(f)
                label = int(file_name.split('_')[-1].split('.')[0])
                labels.append(label)
                images.append(im)
            images, indices = add_similarity_channel(images,images,c,only_sim=True,sim_layer_size=32,random_shuffle=True)
            for i in range(len(images)):
                cv2.imwrite(output_path+'/'+str(labels[indices[i]])+'/'+file_names[indices[i]], images[i])


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
    for tiles_per_dim in [2]:#,4,5]: TODO: change!
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
                    for j in range(len(tiles_in_distribution)):
                        if i == j:
                            continue
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

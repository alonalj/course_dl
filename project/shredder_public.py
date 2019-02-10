import cv2
import os
import numpy as np



def original():
    Xa = []
    Xb = []
    y = []


    #TODO: need this for multiple tile sizes, as well as for documents

    IM_DIR = "images/"
    OUTPUT_DIR = "output/"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    files = os.listdir(IM_DIR)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 4

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


def shred_for_rows_cols(isImg, tiles_per_dim):
    Xa = []
    Xb = []
    y = []

    if isImg:
        IM_DIR = "images/"
    else:
        IM_DIR = "documents/"

    OUTPUT_DIR = "dataset_rows_cols_{}_isImg_{}/".format(tiles_per_dim, IM_DIR == "images/")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    else:
        print("Already shredded")
        return

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


def get_row_col_label(label, t, is_rows):
    label = int(label)
    if t == 2:
        if is_rows:
            if label in [0,1]:
                label = 0
            else:
                label = 1
        else:
            if label in [0,2]:
                label = 0
            else:
                label = 1
    if t == 4:
        if is_rows:
            if label in [0,1,2,3]:
                label = 0
            elif label in [4,5,6,7]:
                label = 1
            elif label in [8,9,10,11]:
                label = 2
            else:  # [ 12,13,14,15]
                label = 3
        else:
            if label in [0,4,8,12]:
                label = 0
            elif label in [1,5,9,13]:
                label = 1
            elif label in [2,6,10,14]:
                label = 2
            else:
                label = 3
    if t == 5:
        if is_rows:
            if label in [0,1,2,3,4]:
                label = 0
            elif label in [5,6,7,8,9]:
                label = 1
            elif label in [10,11,12,13,14]:
                label = 2
            elif label in [15,16,17,18,19]:
                label = 3
            else: # [20,21,22,23,24]
                label = 4
        else:
            if label in [0,5,10,15,20]:
                label = 0
            elif label in [1,6,11,16,21]:
                label = 1
            elif label in [2,7,12,17,22]:
                label = 2
            elif label in [3,8,13,18,23]:
                label = 3
            else:
                label = 4
    return label


def create_rows_cols_folders_by_class(tiles_per_dim, isImg, rows_or_cols):
    import shutil

    Xa = []
    Xb = []
    y = []


    OUTPUT_DIR_TRAIN = rows_or_cols + "_" + str(tiles_per_dim) + "/"
    OUTPUT_DIR_TEST = rows_or_cols + "_" + str(tiles_per_dim) + "_val/"
    if not os.path.exists(OUTPUT_DIR_TRAIN):
        os.mkdir(OUTPUT_DIR_TRAIN)
        os.mkdir(OUTPUT_DIR_TEST)
    else:
        print("folders already created.")
        return

    # for rows_or_cols in ["rows", "cols"]:
    IM_DIR = "dataset_rows_cols_{}_isImg_{}/".format(tiles_per_dim, isImg)
    files = np.array(os.listdir(IM_DIR))
    np.random.shuffle(files)
    files_train = files[:int(len(files)*0.8)]

    for f in os.listdir(IM_DIR):
        label = int(f.split('_')[-1].split('.')[0])
        # label = get_row_col_label(label, tiles_per_dim, rows_or_cols == "rows")
        label = str(label)
        if not os.path.exists(OUTPUT_DIR_TRAIN+label):
            os.mkdir(OUTPUT_DIR_TRAIN+label)
            os.mkdir(OUTPUT_DIR_TEST+label)
        if f in files_train:
            OUTPUT_DIR = OUTPUT_DIR_TRAIN
        else:
            OUTPUT_DIR = OUTPUT_DIR_TEST

        shutil.copy(IM_DIR+f,OUTPUT_DIR+label+"/")



# shred_for_img_vs_doc()
# original()


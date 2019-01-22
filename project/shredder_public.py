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
    files = os.listdir(IM_DIR)

    # update this number for 4X4 crop 2X2 or 5X5 crops.
    tiles_per_dim = 2

    for f in files[:1]:
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



shred_for_img_vs_doc()


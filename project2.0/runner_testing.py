from conf import Conf

from datetime import datetime
import time
from preprocessor import *


# TODO change name to testing shredder

base_max_size = 112

for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [True]:
        for tiles_per_dim in [2]:  # [4]: #
            # if not is_images and tiles_per_dim < 5:
            #     continue

            max_size = base_max_size

            data_split_dict = "files_train_img_{}".format(is_images)

            c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
            print(c.n_tiles_per_sample)
            output_dir = "dataset_test_{}_isImg_{}/".format(c.tiles_per_dim,is_images)
            shredder_original(is_images,tiles_per_dim,c,output_dir,dict=data_split_dict)            # run(c)

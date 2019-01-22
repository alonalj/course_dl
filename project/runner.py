from conf import Conf


from datetime import datetime
from model import *

import time
from preprocessor import *

base_max_size = 32
sample = False

if sample:
    for i in range(1):  # for majority vote
        mID = str(i) + "_" + str(time.time())
        for is_images in [False]:
            for tiles_per_dim in [2]:  # [4]: #
                # if not is_images and tiles_per_dim < 5:
                #     continue
                if tiles_per_dim == 2:
                    max_size = base_max_size
                elif tiles_per_dim == 4:
                    max_size = 32  # base_max_size // 2
                else:
                    max_size = 32  # base_max_size // 2
                if is_images:
                    data_split_dict = "train_test_val_dict_img_{}".format(tiles_per_dim)
                else:
                    data_split_dict = "train_test_val_dict_doc_{}".format(tiles_per_dim)
                c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
                print(c.n_tiles_per_sample)
                run_shredder(c)  # will run only if no folder with name of shredded files already exists
                run(c)

else:
    for i in range(3):  # for majority vote
        mID = str(i)+"_"+str(time.time())
        for is_images in [True]: #[True]:  #
            for tiles_per_dim in [4]: # [4]: #
                if tiles_per_dim == 2:
                    max_size = base_max_size
                elif tiles_per_dim == 4:
                    max_size = base_max_size #base_max_size // 2
                else:
                    max_size = 16 #base_max_size // 2
                if is_images:
                    data_split_dict = "train_test_val_dict_img_{}".format(tiles_per_dim)
                else:
                    data_split_dict = "train_test_val_dict_doc_{}".format(tiles_per_dim)
                c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
                print(c.n_tiles_per_sample)
                run_shredder(c)  # will run only if no folder with name of shredded files already exists
                run(c)
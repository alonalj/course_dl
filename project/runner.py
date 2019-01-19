from conf import Conf


from datetime import datetime
from model import *

import time
for i in range(1):  # for majority vote
    mID = str(time.time())
    for is_images in [False, True]:
        for tiles_per_dim in [2]:
            tiles_per_dim = 2
            max_size = 56
            if is_images:
                data_split_dict = "train_test_val_dict_img"
            else:
                data_split_dict = "train_test_val_dict_doc"
            c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
            print(c.n_tiles_per_sample)
            run(c)
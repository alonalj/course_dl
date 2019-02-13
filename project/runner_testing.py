from conf import Conf


from datetime import datetime
from model import *

import time
from preprocessor import *


# TODO change name to testing shredder

base_max_size = 112

for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [False]:
        for tiles_per_dim in [4]:  # [4]: #
            # if not is_images and tiles_per_dim < 5:
            #     continue

            max_size = base_max_size

            data_split_dict = "files_train_img_{}".format(is_images)

            c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
            print(c.n_tiles_per_sample)
            run_shredder(c)  # will run only if no folder with name of shredded files already exists
            # run(c)

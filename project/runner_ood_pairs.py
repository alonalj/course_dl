from conf import Conf


from datetime import datetime
from model_ood_pairs import *

import time
from preprocessor import *

base_max_size = 64


for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [True]:
        for tiles_per_dim in [4]:  # [4]: #
            # if not is_images and tiles_per_dim < 5:
            #     continue
            if tiles_per_dim == 2:
                max_size = base_max_size
            elif tiles_per_dim == 4:
                max_size = 64  # base_max_size // 2
            else:
                max_size = 64  # base_max_size // 2
            if is_images:
                data_split_dict = "train_test_val_dict_img_{}".format(tiles_per_dim)
            else:
                data_split_dict = "train_test_val_dict_doc_{}".format(tiles_per_dim)
            c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
            print(c.n_tiles_per_sample)
            # TODO call folder creator
            shred_for_ood_pairs(True)
            create_ood_non_ood_pairs(True)
            run(c)

from conf import Conf


from datetime import datetime
from model_ood_pairs import *

import time
from preprocessor import *

base_max_size = 64


for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [True,False]:
        # if not is_images and tiles_per_dim < 5:
        #     continue
        c = Conf(0, base_max_size, is_images, mID)
        print(c.n_tiles_per_sample)
        # TODO call folder creator
        shred_for_ood_pairs(is_images)
        create_ood_non_ood_pairs(is_images)
        run(c)

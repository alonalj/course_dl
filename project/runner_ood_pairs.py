from conf import Conf


from datetime import datetime
from resnet_ood_pairs_folder_based import run

import time
from preprocessor import *

base_max_size = 64


for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [False]:
        # if not is_images and tiles_per_dim < 5:
        #     continue
        c = Conf()
        c.max_size = base_max_size
        c.is_images = is_images
        shred_for_ood_pairs(is_images)
        create_ood_non_ood_pairs(c)
        run(c)

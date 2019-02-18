# from resnet_ood_classifier import run
from model_ood_pairs import run
import time
from preprocessor import *

base_max_size = 112


for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [True]:
        c = Conf()
        c.max_size = base_max_size
        c.is_images = is_images
        shred_for_ood_pairs(is_images)
        create_ood_non_ood_pairs(c)
        run(c)

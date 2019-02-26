# from resnet_ood_classifier import run
from model_ood_pairs import run
import time
from preprocessor import *

base_max_size = 32


for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [True]:
        c = Conf()
        c.max_size = base_max_size
        c.is_images = is_images
        c.tiles_per_dim = 5
        c.data_split_dict = False
        shredder_with_oods('documents/', c, 'shredded_oods_{}/'.format(c.tiles_per_dim))
        # shred_for_ood_pairs(is_images)
        # create_ood_non_ood_pairs(c)
        # create_ood_files_by_similarity('shredded_oods_{}/'.format(c.tiles_per_dim),'ood_isImg_{}'.format(c.is_images),c)
        run(c)

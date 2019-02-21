from conf import Conf

from datetime import datetime
import time
from preprocessor import *
from evaluate import evaluate_internal
import glob

base_max_size = 112

res = {}
for i in range(1):  # for majority vote
    mID = str(i) + "_" + str(time.time())
    for is_images in [False]:
        for tiles_per_dim in [4]:  # [4]: #
            # if not is_images and tiles_per_dim < 5:
            #     continue

            max_size = base_max_size

            data_split_dict = "files_test_img_{}".format(is_images)

            c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
            print(c.n_tiles_per_sample)
            output_dir = "dataset_test_{}_isImg_{}/".format(c.tiles_per_dim,is_images)

            # EVAL WITHOUT OODS
            # shredder_original(is_images,tiles_per_dim,c,output_dir,dict=data_split_dict)            # run(c)

            # EVAL WITH OODS
            shredder_with_oods('images/' if is_images else 'documents/', c, 'shredded_oods_{}/'.format(c.tiles_per_dim))

            tiles_per_dim = 4
            is_img = False
            list_f = glob.glob('shredded_oods_{}/*'.format(tiles_per_dim))
            list_f = sorted(list_f)
            n_tiles_total = tiles_per_dim ** 2

            all_same = []
            for folder in list_f:
                files = glob.glob(folder + '/*')
                c = Conf()
                c.max_size = 112
                c.tiles_per_dim = tiles_per_dim
                c.is_images = is_img
                preds = evaluate_internal(c, files, False)
                actual = list(range(c.tiles_per_dim ** 2))
                oods = [-1 for i in range(len(preds) - c.tiles_per_dim ** 2)]
                actual.extend(oods)

                same = [1 if preds[i] == actual[i] else 0 for i in range(len(preds))]
                accuracy = np.mean(same)
                all_same.extend(same)
                accuracy_ongoing = np.mean(all_same)
                print("current accuracy", accuracy)
                print("ongoing accuracy", accuracy_ongoing)
            res[(is_images,tiles_per_dim)] = accuracy_ongoing

print("final test set results: ", res)



from conf import Conf

from datetime import datetime
import time
from preprocessor import *
from evaluate_internal_quick import evaluate_internal, get_rows_cols_model
import glob
import shutil


def get_actuals(files):
    actuals = []
    for f in files:
        ood_label = int(f.split('_')[-1].split('.')[0])
        if ood_label == 1:
            actuals.append(-1)
        else:
            label = int(f.split('_')[-2])
            actuals.append(label)
    print("f", files)
    print("actuals", actuals)
    return actuals

base_max_size = 112

res = {}
for i in range(1):  # in case we want to use majority vote later
    mID = str(i) + "_" + str(time.time())
    for is_images in [False, True]:
        for tiles_per_dim in [2,4,5]:

            max_size = base_max_size

            data_split_dict = "files_test_img_{}".format(is_images)

            c = Conf(tiles_per_dim, max_size, is_images, mID, data_split_dict)
            print(c.n_tiles_per_sample)
            output_dir = "dataset_test_{}_isImg_{}/".format(c.tiles_per_dim,is_images)

            # EVAL WITHOUT OODS
            shredder_with_oods('images/' if is_images else 'documents/', c, 'shredded_oods_{}/'.format(c.tiles_per_dim), add_t_OoDs=False)

            # EVAL WITH OODS
            # shredder_with_oods('images/' if is_images else 'documents/', c, 'shredded_oods_{}/'.format(c.tiles_per_dim), add_t_OoDs=True)

            is_img = is_images
            list_f = glob.glob('shredded_oods_{}/*'.format(tiles_per_dim))
            list_f = sorted(list_f)
            n_tiles_total = tiles_per_dim ** 2

            model_rows = get_rows_cols_model(c, is_rows=True)
            model_cols = get_rows_cols_model(c, is_rows=False)

            all_same = []
            accuracy_ongoing = 0
            acc_count = 0
            for folder in list_f:
                acc_count += 1
                files = glob.glob(folder + '/*')
                files.sort()
                c = Conf()
                c.max_size = 112
                c.tiles_per_dim = tiles_per_dim
                c.is_images = is_img
                actual = get_actuals(files)
                preds = evaluate_internal(c, files, model_rows=model_rows, model_cols=model_cols)
                # actual = list(range(c.tiles_per_dim ** 2))
                # oods = [-1 for i in range(len(preds) - c.tiles_per_dim ** 2)]
                # actual.extend(oods)

                same = [1 if preds[i] == actual[i] else 0 for i in range(len(preds))]
                accuracy = np.mean(same)
                # all_same.extend(same)
                accuracy_ongoing += accuracy
                print("current accuracy", accuracy)
                print("ongoing accuracy", accuracy_ongoing / float(acc_count))
                print('\n')
            res[(is_images,tiles_per_dim)] = accuracy_ongoing / float(acc_count)
            shutil.rmtree('shredded_oods_{}/'.format(tiles_per_dim))

print("final test set results: ", res)



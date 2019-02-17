from conf import Conf
from preprocessor import split_train_val_test, create_normalization_stats, shred_with_similarity_channel
from resnet_rows_cols_classifier import run

for i in range(1):  # for majority vote
    # mID = str(i) + "_" + str(time.time())
    for is_images in [True, False]:
        for tiles_per_dim in [2,4,5]:#,5,2]:  # [4]: #

            # for rows_or_cols in :
            # if tiles_per_dim == 5:
            #     rows_or_cols_list = ["rows"]
            # else:
            #     rows_or_cols_list = ["cols"]
            for rows_or_cols in ["rows","cols"]:#, "cols"]:
                # if is_images and tiles_per_dim == 4 and rows_or_cols == "rows":  # TODO 4, cols
                #     continue
                print("Training for: is_images", is_images, tiles_per_dim, rows_or_cols)
                c = Conf(tiles_per_dim=tiles_per_dim, max_size=112, is_images=is_images)
                OUTPUT_DIR = "dataset_rows_cols_{}_isImg_{}/".format(tiles_per_dim, is_images)
                c.output_dir = OUTPUT_DIR
                shred_with_similarity_channel(is_images, tiles_per_dim, c, OUTPUT_DIR)
                split_train_val_test(is_images)
                create_normalization_stats(c, rows_or_cols)
                run(c, rows_or_cols)

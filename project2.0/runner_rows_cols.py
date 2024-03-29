from conf import Conf
from preprocessor import split_train_val_test, create_normalization_stats, shred_with_similarity_channel
from resnet_rows_cols_classifier import run

for i in range(1):
    for is_images in [True]:
        for rows_or_cols in ["rows","cols"]:
            for tiles_per_dim in [5, 4]:
                print("Training for: is_images", is_images, tiles_per_dim, rows_or_cols)
                c = Conf(tiles_per_dim=tiles_per_dim, max_size=112, is_images=is_images)
                OUTPUT_DIR = "dataset_rows_cols_{}_isImg_{}/".format(tiles_per_dim, is_images)
                c.output_dir = OUTPUT_DIR
                shred_with_similarity_channel(is_images, tiles_per_dim, c, OUTPUT_DIR, add_sim=False)
                split_train_val_test(is_images)
                create_normalization_stats(c, rows_or_cols)
                run(c, rows_or_cols)

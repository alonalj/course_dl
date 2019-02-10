from shredder_public import shred_for_rows_cols, create_rows_cols_folders_by_class
from resnet_rows_cols_folder_based import *

for i in range(1):  # for majority vote
    # mID = str(i) + "_" + str(time.time())
    for is_images in [True]:
        for tiles_per_dim in [4]:  # [4]: #
            for rows_or_cols in ["rows", "cols"]:
                c = Conf(tiles_per_dim, None, is_images)
                print(c.n_tiles_per_sample)
                shred_for_rows_cols(is_images, tiles_per_dim)
                create_rows_cols_folders_by_class(tiles_per_dim, is_images, rows_or_cols)
                run(c, rows_or_cols)

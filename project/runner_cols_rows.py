from preprocessor import shred_for_rows_cols, create_rows_cols_folders_by_class
from resnet_rows_cols_folder_based import *

for i in range(1):  # for majority vote
    # mID = str(i) + "_" + str(time.time())
    for is_images in [False]:
        for tiles_per_dim in [4,5]:  # [4]: #
            # for rows_or_cols in :
            # if tiles_per_dim == 5:
            #     rows_or_cols_list = ["rows"]
            # else:
            #     rows_or_cols_list = ["cols"]
            for rows_or_cols in ["rows", "cols"]:
                c = Conf(tiles_per_dim=tiles_per_dim, max_size=25, is_images=is_images)
                print(c.n_tiles_per_sample)
                shred_for_rows_cols(is_images, tiles_per_dim, c)
                create_rows_cols_folders_by_class(tiles_per_dim, is_images, rows_or_cols)
                run(c, rows_or_cols)

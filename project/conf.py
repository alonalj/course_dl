class Conf:

    tiles_per_dim = 2
    n_original_tiles = tiles_per_dim ** 2
    n_tiles_per_sample = n_original_tiles + tiles_per_dim
    n_classes = n_original_tiles + 1  # (e.g. for t=2 we have 0,1,2,3 originals and 1 OoD who will get the label 4)
    max_size = 56 // 2


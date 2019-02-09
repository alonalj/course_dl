class Conf:

    def __init__(self, tiles_per_dim=2, max_size=32, is_images=True, mID="", data_split_dict=""):
        self.tiles_per_dim = int(tiles_per_dim)
        self.n_original_tiles = int(tiles_per_dim ** 2)
        self.n_tiles_per_sample = int(self.n_original_tiles + tiles_per_dim)
        self.n_classes = int(self.n_original_tiles + 1)  # (e.g. for t=2 we have 0,1,2,3 originals and 1 OoD who will get the label 4)
        self.max_size = int(max_size)
        self.is_images = is_images
        self.mID = mID
        self.data_split_dict = data_split_dict
        self.is_sample = True

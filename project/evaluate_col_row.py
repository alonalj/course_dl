import os
import cv2
from keras.utils import to_categorical
from preprocessor import *
from resnet_rows_cols_folder_based import *
from resnet_ood import *
from resnet_img_doc_classifier import *


def maybe_download_weights():
    import os
    try:
        import urllib.request
    except:
        print("****** urllib package not installed - cannot fetch the solutions.******")
        return
    try:
        # Resulting weights for section "Fine tunning"
        if not os.path.exists('resnet_maxSize_32_t_5_isImg_False.h5'):
            print("Downloading our solution weight files. This may take a few minutes (315 MB total).")
            urllib.request.urlretrieve("https://drive.google.com/uc?id=1zr16MMaMdYe06D_YiPqkSVCcCCU93MS1&authuser=0&export=download",
                                       'resnet_maxSize_32_t_5_isImg_False.h5')
        if not os.path.exists('resnet_maxSize_32_t_5_isImg_True.h5'):
            urllib.request.urlretrieve("https://drive.google.com/uc?id=1xnJUYvlCmi86BGMcgdSC_Bhi-0vwq2ht&authuser=0&export=download",
                                       'resnet_maxSize_32_t_5_isImg_True.h5')
        if not os.path.exists('resnet_maxSize_32_t_4_isImg_False.h5'):
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?id=17ND6soRS86zmct1SxY8UwxO2vSm9D2FL&authuser=0&export=download",
                'resnet_maxSize_32_t_4_isImg_False.h5')
        if not os.path.exists('resnet_maxSize_32_t_4_isImg_True.h5'):
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?id=1fk--OsWqIp9JjLwoBcme0RgwrEAuhhnA&authuser=0&export=download",
                'resnet_maxSize_32_t_4_isImg_True.h5')
        if not os.path.exists('resnet_maxSize_32_t_2_isImg_False.h5'):
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?id=12fcuqkor0coPUdc5xmlE-J7NXNCJ4lzV&authuser=0&export=download",
                'resnet_maxSize_32_t_2_isImg_False.h5')
        if not os.path.exists('resnet_maxSize_32_t_2_isImg_True.h5'):
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?id=1QJuc_FLmjshPcJMiNLjs_ygrHoOr3VlD&authuser=0&export=download",
                'resnet_maxSize_32_t_2_isImg_True.h5')
        if not os.path.exists('is_img_or_doc.h5'):
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?id=1-Vdbo7QXWVkXy4UsTkoErDGaDoIaiCKw&authuser=0&export=download",
                'is_img_or_doc.h5')
            print("Completed weight downloads.")
        return
    except:
        print("****** \nCannot auto-download the solution weights.\nPlease use the following link to download manually: "
              "https://drive.google.com/open?id=1aYCefWtPdV06L7dlHxDsdC_jRfWgf3j0 \n******")
        return


def get_t(images):
    n_images = len(images)
    if n_images <= 2**2+2:
        return 2
    if n_images <= 4**2+4:
        return 4
    return 5


def get_image_size(t):
    # if t == 2 or t == 4:
    return 32


def is_image(images):
    images = np.array(images)
    resnet_img_vs_doc = build_resnet_img_vs_doc(1e-3)
    resnet_img_vs_doc.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    resnet_img_vs_doc.load_weights('is_img_or_doc.h5')
    resized_images = []
    for im in images:
        resized_images.append(cv2.resize(im, (32, 32)).reshape((32, 32, 1)))#.reshape((32,32,1)))
    # resnet_img_vs_doc.predict_on_batch(np.array([cv2.resize(im, (32, 32)).reshape((32, 32, 1))]))
    resized_images = np.array(resized_images)
    preds = resnet_img_vs_doc.predict_on_batch(resized_images)
    preds = preds.argmax(axis=1)
    if np.median(preds) == 1:
        return True
    return False


def get_ood_model(c, is_image):
    resnet_ood = build_resnet_ood(c.max_size, c.n_tiles_per_sample, 2, c.n_original_tiles, c.tiles_per_dim)
    resnet_ood.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    if is_image:
        resnet_ood.load_weights('ood_resnet_maxSize_32_tilesPerDim_4_nTilesPerSample_20_isImg_True_mID_0_1549712345.601242_L_0.8117778.h5')
    return resnet_ood


def get_rows_cols_model(c):
    resnet = build_resnet_rows_col(c.tiles_per_dim)

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])
    return resnet


def predict(images):
    # maybe_download_weights()  # no need - function called in  weights_download
    labels = []
    X_batch = []
    t = get_t(images)
    img_ix_to_label_rows = {}
    img_ix_to_label_cols = {}

    isImg = is_image(images)

    conf_ood = Conf(int(t), 32, isImg)
    conf_row_col = Conf(int(t), 112, isImg)

    # OOD classification
    resnet_ood = get_ood_model(conf_ood, isImg)

    resized_images_for_ood = []
    resized_images_for_col_row = []
    original_images_for_ood = []
    original_images_for_col_row = []
    add_n_null_images = 0
    n_expected_oods = len(images) - conf_ood.n_original_tiles
    if len(images) < conf_ood.n_tiles_per_sample:
        add_n_null_images = conf_ood.n_tiles_per_sample - len(images)
    for im in images:
        original_images_for_ood.append(im)
        original_images_for_col_row.append(im)

        resized_images_for_ood.append(resize_image(im, conf_ood.max_size,
                                                   simple_reshape=True))  # TODO: has to be here! Only after determining doc / image
        resized_images_for_col_row.append(resize_image(im, conf_ood.max_size,
                                                       simple_reshape=True))

    # adding additional OoD images (null image) to reach t OoDs per sampled folder
    for n_null in range(add_n_null_images):
        resized_images_for_ood.append(np.zeros((conf_ood.max_size, conf_ood.max_size)))
        original_images_for_ood.append(np.zeros((conf_ood.max_size, conf_ood.max_size)))
    resized_images_for_ood = add_similarity_channel(resized_images_for_ood, original_images_for_ood, conf_ood)

    X_batch.append(np.array(resized_images_for_ood))  # a folder is one single sample
    X_batch = list(np.array(X_batch).reshape(conf_ood.n_tiles_per_sample, 1, conf_ood.max_size, conf_ood.max_size, 2))

    logits_ood = resnet_ood.predict_on_batch(X_batch)[:len(images)]
    print(logits_ood)
    logits_ood_with_idx = zip(logits_ood, range(len(logits_ood)))
    logits_ood_with_idx = sorted(logits_ood_with_idx, key=lambda item: item[0][0]) # TODO: first class is?
    ood_images_idx = [item[1] for item in logits_ood_with_idx[:n_expected_oods]]
    real_images_idx = [item[1] for item in logits_ood_with_idx[n_expected_oods:]]
    print("ood images idx", ood_images_idx)
    for i in ood_images_idx:
        img_ix_to_label_rows[i] = -1

    rows_cols_model = get_rows_cols_model(conf_row_col, isImg)

    # rows
    row_model = rows_cols_model.load_weights('model_{}_{}_isImg_{}.h5'.format("rows", conf_row_col.tiles_per_dim, is_image))
    images_for_row_col_prediction = [images[i] for i in real_images_idx]
    logits_row = row_model.predict_on_batch(images_for_row_col_prediction)
    # greedily start placing those with higher row certainty
    for row in range(conf_row_col.tiles_per_dim):
        relevant_row_logits = [l[row] for l in logits_row]
        logits_non_ood_with_idx = zip(relevant_row_logits, real_images_idx)
        logits_non_ood_with_idx = sorted(logits_non_ood_with_idx, key=lambda item: item[0])
        most_likely_ix_in_row = [item[1] for item in logits_non_ood_with_idx[:conf_row_col.tiles_per_dim]]
        for ix in most_likely_ix_in_row:
            img_ix_to_label_rows[ix] = row

    # cols
    col_model = rows_cols_model.load_weights('model_{}_{}_isImg_{}.h5'.format("cols", conf_row_col.tiles_per_dim, is_image))
    logits_col = col_model.predict_on_batch(images_for_row_col_prediction)
    # greedily start placing those with higher col certainty
    for col in range(conf_row_col.tiles_per_dim):
        relevant_col_logits = [l[col] for l in logits_col]
        logits_non_ood_with_idx = zip(relevant_col_logits, real_images_idx)
        logits_non_ood_with_idx = sorted(logits_non_ood_with_idx, key=lambda item: item[0])
        most_likely_ix_in_col = [item[1] for item in logits_non_ood_with_idx[:conf_row_col.tiles_per_dim]]
        for ix in most_likely_ix_in_col:
            img_ix_to_label_cols[ix] = col
    # print("final labels mapping cols", img_ix_to_label_cols)

    for i in range(len(images)):
        image_row_col_tuple = (img_ix_to_label_rows[i], img_ix_to_label_cols[i])
        label = row_col_tuple_to_position(conf_row_col.tiles_per_dim, image_row_col_tuple)
        labels.append(label)

    # print(labels)

    return labels


def row_col_tuple_to_position(tiles_per_dim, row_col_tuple):
    dictionary = {}
    count = 0
    for i in range(tiles_per_dim):
        for j in range(tiles_per_dim):
            dictionary[(i,j)] = count
            count += 1
    print(dictionary)
    return dictionary[row_col_tuple]


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    print(files)  #TODO: remove
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)


    Y = predict(images)
    print(Y)  # TODO - remove!
    return Y


# # TODO: remove
evaluate('example/')
# evaluate('dataset_4_isImg_True/n01440764_96_crw_45_crh_0_reshape_False/')
# row_col_tuple_to_position(5, (2,2))
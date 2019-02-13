import os
import cv2
from keras.utils import to_categorical
from preprocessor import *
from resnet_rows_cols_folder_based import *
from resnet_ood_pairs_folder_based import build_resnet_ood
from resnet_img_doc_classifier import build_resnet_img_vs_doc


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
    resnet_ood = build_resnet_ood(c.max_size)
    resnet_ood.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    if is_image:
        resnet_ood.load_weights('model_ood_pairs_isImg_{}.h5'.format(c.is_images))
    return resnet_ood


def get_rows_cols_model(c):
    resnet = build_resnet_rows_col(c.tiles_per_dim, c.max_size)

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])
    return resnet


def predict_oods(images, conf_ood, img_ix_to_labels):
    # OOD classification
    resnet_ood = get_ood_model(conf_ood, conf_ood.is_images)

    n_expected_oods = len(images) - conf_ood.n_original_tiles

    all_votes = []
    for i in range(len(images)):
        all_votes_for_image = []
        avg_certainty = 0
        for j in range(len(images)):
            if i == j:
                continue
            im1 = images[i]
            im2 = images[j]
            combined_images = []
            for im in [im1, im2]:
                im = cv2.resize(im, (conf_ood.max_size, conf_ood.max_size))
                combined_images.append(im)
            combined_images = np.concatenate(combined_images, axis=1)
            logits_ood = resnet_ood.predict_on_batch(np.array([np.expand_dims(combined_images,
                                                                              -1)]))  # TODO: change to prediction on batch, not on single example of combined images
            vote = np.argmax(logits_ood, 1)
            avg_certainty += np.mean(logits_ood, 1)[0] / (len(images) - 1)
            all_votes_for_image.append(vote)
        label = np.median(all_votes_for_image)
        all_votes.append((label, avg_certainty, i))
        print(label, avg_certainty)

    # keep only ood votes and add to final labels
    all_votes_ood = [v for v in all_votes if v[0] == 1]
    if len(all_votes_ood) >= n_expected_oods:
        all_votes_ood = sorted(all_votes_ood, key=lambda x: x[1], reverse=True)
        oods = all_votes_ood[:n_expected_oods]
        for o in oods:
            i = o[-1]  # the last index in all_votes = image ix in images
            img_ix_to_labels[i] = -1
        print(img_ix_to_labels)
    else:
        # TODO: add logic for the case where there are no ood votes (label=1) or not enough (sort by logits for position 1 and take highest as oods)
        pass

    # TODO: return image indices for row col preds [non-ood] (sorted in order as they appear in images)
    non_ood_imgs_ix = None
    return img_ix_to_labels, non_ood_imgs_ix


def predict(images, labels_gt=None):

    # maybe_download_weights()  # no need - function called in  weights_download
    print(labels_gt)

    labels = []
    X_batch = []
    t = get_t(images)
    img_ix_to_labels = {}
    img_ix_to_label_cols = {}

    isImg = is_image(images)

    conf_ood = Conf(int(t), 112, isImg)
    conf_row_col = Conf(int(t), 112, isImg)

    # img_ix_to_labels, non_ood_images_ix = predict_oods(images, conf_ood, img_ix_to_labels)
    non_ood_images_ix = -1 # TODO remove
    img_ix_to_labels_rows = predict_rows_cols(images, non_ood_images_ix, conf_row_col, labels_gt, is_rows=True)
    # img_ix_to_labels_cols = predict_rows_cols(images, non_ood_images_ix, conf_row_col, labels_gt, is_rows=False)

    # convert (row,col) tuple to position and add to mapping from img_ix to label
    # get final list of labels
    # for i in range(len(images)):
    #     image_row_col_tuple = (img_ix_to_labels_rows[i], img_ix_to_labels_cols[i])
    #     label = row_col_tuple_to_position(conf_row_col.tiles_per_dim, image_row_col_tuple) #TODO
    #     labels.append(label)
    #
    # print(labels)
    # return labels

def predict_rows_cols(images, non_ood_images_ix, conf_row_col, labels_gt=None, is_rows=True):
    '''
    :param images: images without oods
    :param conf_row_col:
    :return:
    '''

    non_ood_images = []
    if non_ood_images_ix == -1:
        non_ood_images = [cv2.resize(im, (conf_row_col.max_size, conf_row_col.max_size)).reshape((conf_row_col.max_size, conf_row_col.max_size, 1)) for im in images]
        non_ood_images_ix = range(len(images))
    else:
        for i in non_ood_images_ix:
            non_ood_images.append(cv2.resize(images[i], (conf_row_col.max_size, conf_row_col.max_size)).reshape((conf_row_col.max_size, conf_row_col.max_size, 1)))

    # predicting rows and cols
    rows_cols_model = get_rows_cols_model(conf_row_col)

    model_type = "rows" if is_rows else "cols"
    print(model_type)
    # rows_cols_model = keras.models.load_model('model_net_{}_{}_isImg_{}.h5'.format(model_type, conf_row_col.tiles_per_dim, conf_row_col.is_images))
    rows_cols_model.load_weights('model_weights_{}_{}_isImg_{}.h5'.format(model_type, conf_row_col.tiles_per_dim, conf_row_col.is_images))
    # non_ood_images = add_similarity_channel(non_ood_images, non_ood_images, conf_row_col, sim_on_side=True)
    # resized_images = []
    # for im in non_ood_images:
    #     resized_images.append(np.expand_dims(im,-1))
    # non_ood_images = resized_images
    logits = rows_cols_model.predict_on_batch(np.array(non_ood_images))
    print(np.argmax(logits,1))
    logits_img_ix_pos_tuples = []
    argmax_preds = []
    for im_ix_internal in range(len(logits)):
        # print(logits[im_ix_internal])
        argmax_preds.append(np.argmax(logits[im_ix_internal]))
        for pos in range(len(logits[im_ix_internal])):
            score = logits[im_ix_internal][pos]
            logits_img_ix_pos_tuples.append((score, im_ix_internal, pos))
    pos_list_sorted = sorted(logits_img_ix_pos_tuples, key=lambda x: x[0], reverse=True)
    internal_im_ix_to_pos = {}
    count_pos_to_allocated_imgs = {}
    img_ix_allocated = []
    for pos in range(conf_row_col.tiles_per_dim):
        count_pos_to_allocated_imgs[pos] = 0
    for tuple in pos_list_sorted:
        score, im_ix_internal, pos = tuple
        # print(pos)
        # print(count_pos_to_allocated_imgs)
        if count_pos_to_allocated_imgs[pos] >= conf_row_col.tiles_per_dim or im_ix_internal in img_ix_allocated:
            continue
        internal_im_ix_to_pos[im_ix_internal] = pos
        img_ix_allocated.append(im_ix_internal)
        count_pos_to_allocated_imgs[pos] += 1

    final_im_ix_to_pos = {}
    for im_ix_internal in internal_im_ix_to_pos.keys():
        img_ix_global = non_ood_images_ix[im_ix_internal]
        final_im_ix_to_pos[img_ix_global] = internal_im_ix_to_pos[im_ix_internal]

    # return final_im_ix_to_pos

    pos_preds = []
    for i in non_ood_images_ix:
        pos_preds.append(final_im_ix_to_pos[i])
    print(pos_preds)
    n_correct = len([1 for i in range(len(pos_preds)) if pos_preds[i] == labels_gt[i]])
    print("greedy preds", n_correct/len(pos_preds))
    n_correct = len([1 for i in range(len(argmax_preds)) if argmax_preds[i] == labels_gt[i]])
    print("argmax preds", n_correct / len(pos_preds))

    #     # top_img_tuples_for_pos = pos_list_sorted[:conf_row_col.tiles_per_dim]
    #     for img_tuple in pos_list_sorted:
    #         im_ix_internal = img_tuple[1]
    #         if im_ix_internal not in placed_image_ix_internal
    #         internal_im_ix_to_pos[im_ix_internal] = pos
    #         placed_image_ix_internal.append(im_ix_internal)
    # final_im_ix_to_pos = {}
    # for im_ix_internal in internal_im_ix_to_pos.keys():
    #     img_ix_global = non_ood_images_ix[im_ix_internal]
    #     final_im_ix_to_pos[img_ix_global] = internal_im_ix_to_pos[im_ix_internal]
    # print(final_im_ix_to_pos)



    # non_ood_images = add_similarity_channel(non_ood_images, non_ood_images, conf_row_col, n_channels=3)
    # from PIL import Image

    # datagen_img_vs_doc = ImageDataGenerator(
    #     preprocessing_function=lambda x: x / 255.)  # preprocessing_function=to_grayscale)
    # for im in non_ood_images:
        # im_ = Image.fromarray(im.astype('uint8'))
        # im = np.array(im_)
        # cv2.imwrite("tmp.jpg", im)
        # im = cv2.imread("tmp.jpg")
        # from keras_preprocessing import image

        # # from PIL import Image
        # # im_ = Image.fromarray(im)
        # # im_ = im_.resize((conf_row_col.max_size, conf_row_col.max_size))
        # # im = np.array(im_)
        # im = im / 255.
        # resized_images.append(im)
        # resized_images.append(im / 255.)

    # non_ood_images = resized_images
    # flower = datagen_img_vs_doc.flow(np.array(resized_images))
    # flower.target_size = (conf_row_col.max_size, conf_row_col.max_size)
    # logits = rows_cols_model.predict_generator(flower,steps=1)

    # greedily start placing those with higher row certainty

    # for row in range(conf_row_col.tiles_per_dim):
    #     relevant_row_logits = [l[row] for l in logits]
    #     logits_non_ood_with_idx = zip(relevant_row_logits, real_images_idx)
    #     logits_non_ood_with_idx = sorted(logits_non_ood_with_idx, key=lambda item: item[0])
    #     most_likely_ix_in_row = [item[1] for item in logits_non_ood_with_idx[:conf_row_col.tiles_per_dim]]
    #     for ix in most_likely_ix_in_row:
    #         img_ix_to_label_rows[ix] = row

    # # cols
    # col_model = rows_cols_model.load_weights('model_cols_{}_isImg_{}.h5'.format(conf_row_col.tiles_per_dim, is_image))
    # logits_col = col_model.predict_on_batch(images_for_row_col_prediction)
    # # greedily start placing those with higher col certainty
    # for col in range(conf_row_col.tiles_per_dim):
    #     relevant_col_logits = [l[col] for l in logits_col]
    #     logits_non_ood_with_idx = zip(relevant_col_logits, real_images_idx)
    #     logits_non_ood_with_idx = sorted(logits_non_ood_with_idx, key=lambda item: item[0])
    #     most_likely_ix_in_col = [item[1] for item in logits_non_ood_with_idx[:conf_row_col.tiles_per_dim]]
    #     for ix in most_likely_ix_in_col:
    #         img_ix_to_label_cols[ix] = col
    # # print("final labels mapping cols", img_ix_to_label_cols)


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


def evaluate_internal(tiles_per_dim, file_dir='example/', is_img=True):
    from preprocessor import get_row_col_label
    files = os.listdir(file_dir)
    files.sort()
    # random.shuffle(files)
    print(files)  #TODO: remove
    labels = []
    # if is_img:
    for f in files:
        if is_img:
            label_original = int(f.split('.')[1].split('_')[-1])
        else:
            label_original = int(f.split('.')[0].split('_')[-1])
        labels.append(get_row_col_label(label_original,tiles_per_dim,True))
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images, labels)
    print(Y)  # TODO - remove!
    return Y


# # TODO: remove
# evaluate('example/')
import glob
tiles_per_dim = 4
is_img = False
for folder in glob.glob('dataset_{}_isImg_{}/*'.format(tiles_per_dim, is_img)):
    evaluate_internal(tiles_per_dim, folder+'/', is_img)

    # try:
    #     evaluate_internal(tiles_per_dim, folder+'/')
    # except:
    #     print("skipped", folder)
# evaluate('dataset_4_isImg_True/n01440764_416/')

# row_col_tuple_to_position(5, (2,2))
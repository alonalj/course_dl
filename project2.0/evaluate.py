from preprocessor import *
from resnet_ood_classifier import build_resnet_ood
from resnet_img_doc_classifier import build_resnet_img_vs_doc

def data_generator_pred(images, c):
   X_batch = []
   for im in images:
       im = preprocess_image(im, c)
       X_batch.append(im)
   yield np.array(X_batch)


def get_t(images):
    n_images = len(images)
    if n_images <= 2**2+2:
        return 2
    if n_images <= 4**2+4:
        return 4
    return 5


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
        im = cv2.resize(im, (32, 32))
        im = np.expand_dims(im,-1)
        resized_images.append(im)#.reshape((32, 32, 1)))#.reshape((32,32,1)))
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


def get_rows_cols_model(c, is_rows):
    rows_or_cols = "rows" if is_rows else "cols"
    from resnet_rows_cols_classifier import build_model
    weights = 'weights_img_{}_t_{}_{}'.format(c.is_images,c.tiles_per_dim,rows_or_cols)
    # weights = 'weights_img_True_t_4_rows_L0.21_A0.94_val_L0.27_A0.91' #TODO: remove
    weights = 'weights_img_True_t_2_rows_L0.51_A0.94_val_L4.22_A0.75'
    return build_model(c, weights)


def predict_oods(images, conf_ood, img_ix_to_labels, n_expected_oods):
    '''
    Loads the ood model, which outputs for each image in images whether it is OOD or not.
    One option for training would be to take pairs, one in distribution (label 0), one out of distribution (label 1),
    and at prediction time to have each pair tested, and use majority prediction to eliminate OODs (for example,
    predict for all possible pairs {im_i, im_j}, j!=i  while keeping count of the number of pairs for which im_i was
    predicted as OOD. If im_i has the highest such count (i.e. got a prediction of OOD in more pairs than any other),
    remove im_i first. Proceed to remove the next image with the highest count and so on.
    If none were predicted OODs, simply return the full list. The main model will choose those it is most certain of.
    :param images:
    :param conf_ood:
    :param img_ix_to_labels:
    :return: list of indices corresponding to images identified as NON-odd
    '''

    # TODO remove these two
    non_ood_imgs_ix = list(range(len(images)))
    return non_ood_imgs_ix

    # OOD classification
    resnet_ood = get_ood_model(conf_ood, conf_ood.is_images)

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
            logits_ood = resnet_ood.predict_on_batch(np.array([np.expand_dims(combined_images,-1)]))  # TODO: change to prediction on batch, not on single example of combined images
            vote = np.argmax(logits_ood, 1)
            avg_certainty += np.mean(logits_ood, 1)[0] / (len(images) - 1)
            all_votes_for_image.append(vote)
        label = np.median(all_votes_for_image)
        all_votes.append((label, avg_certainty, i))
        print(label, avg_certainty)

    non_ood_imgs_ix = list(range(len(images))) # default non_ood_imgs_ix is all images, worst-case the position model will take the most confident ones...
    # keep only ood votes and add to final labels
    all_votes_ood = [v for v in all_votes if v[0] == 1]
    if len(all_votes_ood) > 0: # at least one was identified as ood
        all_votes_ood = sorted(all_votes_ood, key=lambda x: x[1], reverse=True)
        oods = all_votes_ood[:n_expected_oods]
        for o in oods:
            i = o[-1]  # the last index in all_votes = image ix in images
            img_ix_to_labels[i] = -1
        print(img_ix_to_labels)
        non_ood_imgs_ix.pop(i) # TODO remove ix that's OOD

    return list(non_ood_imgs_ix)


def prepare_images(images, c):
    images = add_similarity_channel(images, images, c, sim_on_side=True)
    processed_images = []
    for im in images:
        im = preprocess_image(im, c)
        processed_images.append(im)
    return processed_images


def predict(images, labels_gt=None):

    print(labels_gt)

    labels = []
    t = get_t(images)
    img_ix_to_labels = {}

    isImg = is_image(images)

    conf_ood = Conf(int(t), 112, False)
    conf_row_col = Conf(int(t), 112, isImg)

    n_expected_oods = len(images) - conf_ood.n_original_tiles

    non_ood_images_ix = predict_oods(images, conf_ood, img_ix_to_labels, n_expected_oods)

    img_ix_to_labels_rows = predict_rows_cols(images, non_ood_images_ix, conf_row_col, labels_gt, is_rows=True)
    img_ix_to_labels_cols = predict_rows_cols(images, non_ood_images_ix, conf_row_col, labels_gt, is_rows=False)

    # convert (row,col) tuple to position and add to mapping from img_ix to label
    # get final list of labels TODO: check this when cols finishes training
    for i in range(len(images)):
        image_row_col_tuple = (img_ix_to_labels_rows[i], img_ix_to_labels_cols[i])
        label = row_col_tuple_to_position(conf_row_col.tiles_per_dim, image_row_col_tuple) #TODO
        labels.append(label)

    print(labels)
    return labels


def predict_rows_cols(images, non_ood_images_ix, conf_row_col, labels_gt=None, is_rows=True):
    '''
    :param images: images without oods
    :param conf_row_col:
    :return:
    '''

    non_ood_images = []
    for i in non_ood_images_ix:
        non_ood_images.append(images[i])

    non_ood_images = prepare_images(non_ood_images, conf_row_col)
    rows_cols_model = get_rows_cols_model(conf_row_col, is_rows)
    logits = rows_cols_model.predict_on_batch(np.array(non_ood_images))
    print(np.argmax(logits,1))
    logits_img_ix_pos_tuples = []
    argmax_preds = []

    # Sort predictions by confidence
    for im_ix_internal in range(len(logits)):
        argmax_preds.append(np.argmax(logits[im_ix_internal]))
        for pos in range(len(logits[im_ix_internal])):
            score = logits[im_ix_internal][pos]
            logits_img_ix_pos_tuples.append((score, im_ix_internal, pos))
    pos_list_sorted = sorted(logits_img_ix_pos_tuples, key=lambda x: x[0], reverse=True)

    # Allocate predictions greedily, until all positions are filled
    internal_im_ix_to_pos = {}
    count_pos_to_allocated_imgs = {}
    img_ix_allocated = []
    for pos in range(conf_row_col.tiles_per_dim):
        count_pos_to_allocated_imgs[pos] = 0
    for tuple in pos_list_sorted:
        score, im_ix_internal, pos = tuple
        if count_pos_to_allocated_imgs[pos] >= conf_row_col.tiles_per_dim or im_ix_internal in img_ix_allocated:
            continue
        internal_im_ix_to_pos[im_ix_internal] = pos
        img_ix_allocated.append(im_ix_internal)
        count_pos_to_allocated_imgs[pos] += 1

    # Map indices back to originally supplied indices (in case OODs were successfully removed)
    final_im_ix_to_pos = {}
    for im_ix_internal in internal_im_ix_to_pos.keys():
        img_ix_global = non_ood_images_ix[im_ix_internal]
        final_im_ix_to_pos[img_ix_global] = internal_im_ix_to_pos[im_ix_internal]

    if labels_gt:
        pos_preds = []
        for i in non_ood_images_ix:
            pos_preds.append(final_im_ix_to_pos[i])
        print(pos_preds)
        n_correct = len([1 for i in range(len(pos_preds)) if pos_preds[i] == labels_gt[i]])
        print("greedy preds", n_correct/len(pos_preds))
        n_correct = len([1 for i in range(len(argmax_preds)) if argmax_preds[i] == labels_gt[i]])
        print("argmax preds", n_correct / len(pos_preds))

    return final_im_ix_to_pos
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


def evaluate_internal(c, files, is_rows=True):
    from preprocessor import get_row_col_label
    # files = os.listdir(file_dir)
    files.sort()
    # random.shuffle(files)
    print(files)  #TODO: remove
    labels = []
    # if is_img:
    for f in files:
        labels.append(get_row_col_label(f,c,is_rows))
    images = []
    for f in files:
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images, labels)
    print(Y)  # TODO - remove!
    return Y


# # TODO: remove
# evaluate('example/')
# evaluate_internal('example/')
import glob
tiles_per_dim = 2
is_img = True
list_f = glob.glob('dataset_test_{}_isImg_{}/*'.format(tiles_per_dim, is_img))
list_f = sorted(list_f)
n_tiles_total = tiles_per_dim**2

for i in range(10):
    files = list_f[i*n_tiles_total:(i+1)*n_tiles_total]
    c = Conf()
    c.max_size = 112
    c.tiles_per_dim = tiles_per_dim
    evaluate_internal(c, files, True)
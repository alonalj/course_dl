from preprocessor import *
from resnet_ood_classifier import build_resnet_ood
from resnet_img_doc_classifier import build_resnet_img_vs_doc


def get_t(images):
    n_images = len(images)
    if n_images <= 2*2+2:
        return 2
    if n_images <= 4*4+4:
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
        resized_images.append(im)
    resized_images = np.array(resized_images)
    preds = resnet_img_vs_doc.predict_on_batch(resized_images)
    preds = preds.argmax(axis=1)
    if np.median(preds) == 1:
        return True
    return False


def get_rows_cols_model(c, is_rows):
    rows_or_cols = "rows" if is_rows else "cols"
    from resnet_rows_cols_classifier import build_model
    weights = 'weights_no_sim_img_{}_t_{}_{}.h5'.format(c.is_images,c.tiles_per_dim,rows_or_cols)
    return build_model(c, weights)


def prepare_images(images, c):
    processed_images = []
    for im in images:
        im = preprocess_image(im, c)
        processed_images.append(im)
    return processed_images


def resolve_clashes(pos_to_img_ix, pos_to_img_ix_leftovers, images):
    '''
    :param pos_to_img_ix: mapping position to image after greedy allocation: (i,j):(score,im_ix)
    :param pos_to_img_ix_leftovers: mapping position to image for images not allocated by greedy: (i,j):(score,im_ix)
    :return: pos_to_img_ix adjusted after resolving clashes (any image ix not in this dictionary will be OoD)
    '''
    while len(pos_to_img_ix_leftovers.keys()) > 0:

        pos_to_n_neighbors_non_clashing = []
        for pos in pos_to_img_ix_leftovers.keys():
            # get number of non-clashing neighbors per position (i.e. pos in greedy list, but not in leftovers)
            i,j = pos
            potential_neighbors = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
            count_neighbors_non_clashing = 0
            certain_neighbors = []
            for neighbor in potential_neighbors:
                if neighbor in pos_to_img_ix.keys() and neighbor not in pos_to_img_ix_leftovers.keys():
                    count_neighbors_non_clashing += 1
                    certain_neighbors.append(neighbor)
            pos_to_n_neighbors_non_clashing.append((count_neighbors_non_clashing,pos, certain_neighbors))

        pos_to_n_neighbors_non_clashing = sorted(pos_to_n_neighbors_non_clashing, key=lambda x: x[0], reverse=True)
        pos_with_largest_number_of_certain_neighbors = pos_to_n_neighbors_non_clashing[0][1]
        _, im_ix_clash_1 = pos_to_img_ix[pos_with_largest_number_of_certain_neighbors]
        neighbors = pos_to_n_neighbors_non_clashing[0][2]

        # collecting all candidate image indices for current clashing position in one list
        im_ix_candidates_for_pos = []
        im_ix_candidates_for_pos.append(im_ix_clash_1)
        for _, im_ix_clash_2 in pos_to_img_ix_leftovers[pos_with_largest_number_of_certain_neighbors]:
            im_ix_candidates_for_pos.append(im_ix_clash_2)

        # calculating average edge cosine similarity to the neighbors with no clashes found above
        best_avg_sim = -1
        for im_ix_candidate in im_ix_candidates_for_pos:
            avg_edge_sim = 0
            for pos_neighbor in neighbors:
                im_ix_neighbor = pos_to_img_ix[pos_neighbor][1]
                edge_cosine_sim = calc_cosine_sim_on_single_edge(
                    pos_with_largest_number_of_certain_neighbors, pos_neighbor,
                    im_ix_candidate, im_ix_neighbor,
                    images
                )
                avg_edge_sim += edge_cosine_sim / float(len(neighbors))
            if avg_edge_sim > best_avg_sim:
                best_avg_sim = avg_edge_sim
                winner_im_ix = im_ix_candidate

        # replace winning candidate with existing and remove clashing position from leftovers dict:
        pos_to_img_ix[pos_with_largest_number_of_certain_neighbors] = (None, winner_im_ix)
        del pos_to_img_ix_leftovers[pos_with_largest_number_of_certain_neighbors]

    return pos_to_img_ix


def predict_position_greedy(logits_rows, logits_cols, conf_row_col, labels_gt=False):
    # takes logits, places t^2 greeedily on grid, and the remaining as predicted, returns two mappings (i,j): im_ix
    # (one greedy outcome, the other, remaining images left out after greedy allocation)
    position_to_score_img_ix = {}
    img_ix_to_best_pos = {}
    argmax_preds = []

    # Sort predictions by confidence
    for im_ix in range(len(logits_rows)):
        im_highest_score = 0
        for i in range(conf_row_col.tiles_per_dim):
            for j in range(conf_row_col.tiles_per_dim):
                row_pos_score, col_pos_score = logits_rows[im_ix][i], logits_cols[im_ix][j]
                # determining_pos_score = np.mean([row_pos_score, col_pos_score])
                determining_pos_score = np.mean([row_pos_score, col_pos_score])

                if (i,j) in position_to_score_img_ix.keys():
                    position_to_score_img_ix[(i,j)].append((determining_pos_score, im_ix))
                else:
                    position_to_score_img_ix[(i, j)] = [(determining_pos_score, im_ix)]
                if determining_pos_score > im_highest_score:
                    im_highest_score = determining_pos_score
                    img_best_pos = (i,j)
        img_ix_to_best_pos[im_ix] = (img_best_pos, determining_pos_score)

    # Allocate predictions greedily (each position (i,j) gets the image with the highest average col, row logits),
    # unless the image is highly likely to be an OOD (in which case leave aside - will be considered a clash to be
    # resolved later
    pos_to_im_ix = {}
    img_ix_allocated = []
    for pos in position_to_score_img_ix.keys():
        pos_list_sorted = sorted(position_to_score_img_ix[pos], key=lambda x: x[0], reverse=True)
        top_tuple = pos_list_sorted[0]
        score, im_ix = top_tuple

        while im_ix in img_ix_allocated:
            pos_list_sorted.pop(0)
            top_tuple = pos_list_sorted[0]
            score, im_ix = top_tuple
        pos_to_im_ix[pos] = (score, im_ix)
        img_ix_allocated.append(im_ix)

    # images that were not allocated, get their most likely position prediction, they will compete with those
    # they clash with later
    pos_to_im_ix_leftovers = {}
    for im_ix in range(len(logits_rows)):
        if im_ix not in img_ix_allocated:
            img_best_pos, score = img_ix_to_best_pos[im_ix]
            if img_best_pos in pos_to_im_ix_leftovers.keys():
                pos_to_im_ix_leftovers[img_best_pos].append((score,im_ix))
            else:
                pos_to_im_ix_leftovers[img_best_pos] = [(score, im_ix)]

    return pos_to_im_ix, pos_to_im_ix_leftovers


def predict(images, labels_gt=None):

    labels = []
    t = get_t(images)
    isImg = is_image(images)
    conf_row_col = Conf(int(t), 112, isImg)

    # predict row and column
    logits_rows = predict_rows_cols(images, conf_row_col, labels_gt, is_rows=True)
    logits_cols = predict_rows_cols(images, conf_row_col, labels_gt, is_rows=False)
    # place tiles greedily in predicted (row, column) by logit score
    pos_to_img_ix, pos_to_img_ix_leftovers = predict_position_greedy(logits_rows, logits_cols, conf_row_col,labels_gt)
    # resolve clashes if any exist (with OoDs there are always clashes to resolve)
    pos_to_img_ix = resolve_clashes(pos_to_img_ix, pos_to_img_ix_leftovers, images)

    # reverse the pos-to-image dictionary to image-to-pos for easy labeling below
    img_ix_to_pos = {}
    for pos in pos_to_img_ix.keys():
        img_ix_to_pos[pos_to_img_ix[pos][1]] = pos

    # convert (row,col) tuple to final position (0,...,t^2) and get final list of labels
    for i in range(len(images)):
        if i in img_ix_to_pos.keys():
            # if the image has been placed
            label = row_col_tuple_to_position(conf_row_col.tiles_per_dim, img_ix_to_pos[i])
        else:
            # if the image has been left out after resolving clashes (--> it's OoD)
            label = -1
        labels.append(label)
    return labels


def predict_rows_cols(images, conf_row_col, labels_gt=None, is_rows=True):
    '''
    :param images: images without oods
    :param conf_row_col:
    :return:
    '''

    non_ood_images = prepare_images(images, conf_row_col)
    rows_cols_model = get_rows_cols_model(conf_row_col, is_rows)
    logits = rows_cols_model.predict_on_batch(np.array(non_ood_images))
    return logits


def row_col_tuple_to_position(tiles_per_dim, row_col_tuple):
    dictionary = {}
    count = 0
    for i in range(tiles_per_dim):
        for j in range(tiles_per_dim):
            dictionary[(i,j)] = count
            count += 1
    # print(dictionary)
    return dictionary[row_col_tuple]


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images)
    return Y

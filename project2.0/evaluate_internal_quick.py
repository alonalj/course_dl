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


def get_ood_model(c, is_image):
    from model_ood_pairs import build_model_simple
    resnet_ood = build_model_simple(c,'ood_weights')
    return resnet_ood


def get_rows_cols_model(c, is_rows):
    rows_or_cols = "rows" if is_rows else "cols"
    from resnet_rows_cols_classifier import build_model
    weights = 'weights_no_sim_img_{}_t_{}_{}.h5'.format(c.is_images,c.tiles_per_dim,rows_or_cols)

    return build_model(c, weights)


def find_potential_oods(images, c):
    '''
    Simple idntification of the tiles who are most likely to be the odd ones out - they will be lower in preference for
    out greedy algorithm.
    :param images:
    :param conf_ood:
    :param img_ix_to_labels:
    :return: list of indices corresponding to images identified as NON-odd
    '''

    removed_indices = []
    n_oods = len(images)-c.tiles_per_dim**2
    im_ix_to_max_sim = []
    for i in range(len(images)):
        max_edges = []
        for j in range(len(images)):
            if i == j or i in removed_indices or j in removed_indices:
                continue
            im1 = cv2.resize(images[i], (224, 224))
            im2 = cv2.resize(images[j], (224, 224))
            max_edge = max(calc_cosine_sim_on_edges(im1, im2,disregard_whites=True))
            max_edges.append(max_edge)
        if i in removed_indices:
            continue
        max_edges = sorted(max_edges,reverse=True)
        avg_top_two_edges = np.mean(max_edges[:2])
        # max_sim = max(max_edges)
        im_ix_to_max_sim.append((i,avg_top_two_edges))

    im_ix_to_max_sim = sorted(im_ix_to_max_sim, key=lambda x: x[1], reverse=False)
    potential_oods = im_ix_to_max_sim[:n_oods]
    potential_oods = [i[0] for i in potential_oods]

    return potential_oods


def prepare_images(images, c):
    print(c.tiles_per_dim, c.max_size, c.is_images)
    # images = add_similarity_channel(images, images, c, sim_on_side=True)
    processed_images = []
    for im in images:
        im = preprocess_image(im, c)
        processed_images.append(im)
    return processed_images


def resolve_clashes(pos_to_img_ix, pos_to_img_ix_leftovers, images):
    '''

    :param pos_to_img_ix: mapping position to image after greedy allocation: (i,j):(score,im_ix)
    :param pos_to_img_ix_leftovers: mapping position to image for images not allocated by greedy: (i,j):(score,im_ix)
    :return:
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

        # collecting all canidate image inidices in one list
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
        pos_to_img_ix[pos_with_largest_number_of_certain_neighbors] = (None, winner_im_ix)  # None because score unnecessary
        del pos_to_img_ix_leftovers[pos_with_largest_number_of_certain_neighbors]

    return pos_to_img_ix

def predict_position(logits_rows, logits_cols, conf_row_col, labels_gt=False):
    preds_rows = np.argmax(logits_rows,1)
    preds_cols = np.argmax(logits_cols,1)
    pos_to_im_ix = {}
    pos_to_im_ix_leftovers = {}

    for im_ix in range(len(preds_rows)):
        i,j = preds_rows[im_ix], preds_cols[im_ix]
        mean_score = np.mean([logits_rows[im_ix][i], logits_cols[im_ix][j]])
        if (i,j) in pos_to_im_ix.keys():
            incumbent_score = pos_to_im_ix[(i,j)][0]
        else:
            incumbent_score = 0
        if incumbent_score < mean_score:
            pos_to_im_ix[(i,j)] = (mean_score, im_ix)
        else:
            if (i,j) in pos_to_im_ix_leftovers.keys():
                pos_to_im_ix_leftovers[(i,j)].append((mean_score, im_ix))
            else:
                pos_to_im_ix_leftovers[(i, j)] = [(mean_score, im_ix)]
    return pos_to_im_ix, pos_to_im_ix_leftovers


def predict_position_greedy(logits_rows, logits_cols, conf_row_col, labels_gt=False):
    # takes logits, places t^2 greeedily on grid, and the remaining as predicted, returns two mappings (i,j): im_ix
    # (one greedy outcome, the other, remaining images left out by greedy allocation)
    # print(logit)
    position_to_score_img_ix = {}
    img_ix_to_best_pos = {}
    argmax_preds = []

    # Sort predictions by confidence
    for im_ix in range(len(logits_rows)):
        im_highest_score = 0
        # argmax_preds.append(np.argmax(logits[im_ix]))
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


def predict(images, labels_gt=None, model_rows=False, model_cols=False, c=False):

    print(labels_gt)

    labels = []
    t = get_t(images)
    if c:
        isImg = c.is_images
    else:
        isImg = is_image(images)
    conf_row_col = Conf(int(t), 112, isImg)

    logits_rows = predict_rows_cols(images, conf_row_col, labels_gt, is_rows=True, model=model_rows)
    print("row preds", np.argmax(logits_rows,1))
    logits_cols = predict_rows_cols(images, conf_row_col, labels_gt, is_rows=False, model=model_cols)
    print("col preds", np.argmax(logits_cols,1))
    # pos_to_img_ix, pos_to_img_ix_leftovers = predict_position(logits_rows, logits_cols, conf_row_col, labels_gt)
    # img_ix_potential_oods = []#find_potential_oods(images, conf_row_col)
    pos_to_img_ix, pos_to_img_ix_leftovers = predict_position_greedy(logits_rows, logits_cols, conf_row_col,labels_gt)
    pos_to_img_ix = resolve_clashes(pos_to_img_ix, pos_to_img_ix_leftovers, images) # returns mapping: (im_ix: (i,j) and im_ix:-1
    img_ix_to_pos = {}
    for pos in pos_to_img_ix.keys():
        img_ix_to_pos[pos_to_img_ix[pos][1]] = pos

    # convert (row,col) tuple to position and add get final list of labels
    for i in range(len(images)):
        if i in img_ix_to_pos.keys():
            label = row_col_tuple_to_position(conf_row_col.tiles_per_dim, img_ix_to_pos[i])
        else:
            label = -1
        labels.append(label)
    return labels


def predict_rows_cols(images, conf_row_col, labels_gt=None, is_rows=True, model=False):
    '''
    :param images: images without oods
    :param conf_row_col:
    :return:
    '''
    non_ood_images = prepare_images(images, conf_row_col)

    if model:
        rows_cols_model = model
    else:
        rows_cols_model = get_rows_cols_model(conf_row_col, is_rows)
    logits = rows_cols_model.predict_on_batch(np.array(non_ood_images))
    # preds = np.argmax(logits,1)
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
    print(files)  #TODO: remove
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images)
    print(Y)  # TODO - remove!
    return Y


def evaluate_internal(c, files, is_rows=True, gt_labels=False, model_rows=False, model_cols=False):
    from preprocessor import get_row_col_label
    files.sort()
    print("f incoming", files)
    if gt_labels:
        labels = []
        # if is_img:
        for f in files:
            if 'DS' in f:
                continue
            labels.append(get_row_col_label(f,c,is_rows))
    else:
        labels = False
    images = []
    for f in files:
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images, labels, model_rows,model_cols,c=c)
    print("preds", Y)
    return Y

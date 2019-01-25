import os
import cv2
from keras.utils import to_categorical
from preprocessor import *
from resnet_adapted import *
from conf import Conf
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
    if t == 2 or t == 4:
        return 32
    return 16


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


def read_test_images_docs(file_dir):
    files = os.listdir(file_dir)
    # files.remove('.DS_Store')
    files.sort()
    images = []
    X_batch = []
    y_batch = []
    labels_in_folder = []
    for f in files:
        print(f)
        if f == '.DS_Store':
            continue
        im = cv2.imread(file_dir + f)
        try:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        except:
            continue
        images.append(im)
    return images
    print(is_image(images))



# def predict(images):
def predict(images, y_batch, overall_acc_before, overall_acc_after):
    maybe_download_weights()
    labels = []
    X_batch = []
    t = get_t(images)
    img_size_dst = get_image_size(t)
    c = Conf(int(t), int(img_size_dst), is_image(images))
    isImg = is_image(images)

    resnet = build_resnet(c.max_size, c.n_tiles_per_sample, c.n_classes, c.n_original_tiles, c.tiles_per_dim)

    # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    # sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

    resnet.compile(
        loss='categorical_crossentropy',
        optimizer="adam",
        metrics=['accuracy'])

    if isImg:
        print("is img")
        resnet.load_weights(
            "resnet_maxSize_{}_t_{}_isImg_True.h5".format(c.max_size, t))
    else:
        print("is doc")
        resnet.load_weights(
            "resnet_maxSize_{}_t_{}_isImg_False.h5".format(c.max_size, t))

    resized_images = []
    original_images = []
    im_shapes = {}
    add_n_null_images = 0
    if len(images) < c.n_tiles_per_sample:
        add_n_null_images = c.n_tiles_per_sample - len(images)
    for im in images:
        original_images.append(im)
        if im.shape not in im_shapes:
            im_shapes[im.shape] = 1
        else:
            im_shapes[im.shape] += 1
        resized_images.append(resize_image(im, c.max_size, simple_reshape=True))  # TODO: has to be here! Only after determining doc / image

    # adding additional OoD images to reach t OoDs per sampled folder
    for n_null in range(add_n_null_images):
        resized_images.append(np.zeros((c.max_size, c.max_size)))
        original_images.append(np.zeros((c.max_size, c.max_size)))
    resized_images = add_similarity_channel(resized_images, original_images, c)

    X_batch.append(np.array(resized_images))  # a folder is one single sample
    # print(labels_in_folder)
    # folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
    # y_batch.append(folder_labels)
    # if len(y_batch) == batch_size:
        # print(np.array(X_batch).ndim)
        # print(np.array(X_batch))
        # if np.array(X_batch).shape[1:] != (c.n_tiles_per_sample, c.max_size, c.max_size, 2):
        #     print(folder)
        #     print(np.array(X_batch).shape)
    X_batch = list(np.array(X_batch).reshape(c.n_tiles_per_sample, 1, c.max_size, c.max_size, 2))

    # X_batch.append(np.array(add_similarity_channel(resized_images, original_images, c)))
    # # print(X_batch.shape)
    # X_batch = list(np.array(X_batch).reshape(c.n_tiles_per_sample, 1, c.max_size, c.max_size, 2))
    print(resnet.evaluate(X_batch, y_batch))
    print(resnet.predict_on_batch(X_batch))
    logits = resnet.predict_on_batch(X_batch)
    for l in logits:
        idx_max = l.argmax(axis=1)
        idx_max = int(idx_max)
        if idx_max == c.n_classes - 1:
            # OoD
            idx_max = -1
        labels.append(idx_max)
    print("before ood", labels)

    # in case OoDs are of a different shape, we can easily label them:
    OoD_have_diff_shape = False
    for im_idx in range(len(images)):
        im = images[im_idx]
        if im_shapes[im.shape] <= t:
            labels[im_idx] = -1
            OoD_have_diff_shape = True

    print("after ood", labels)

    labels = labels[:len(images)]

    gt = [np.argmax(i) if np.argmax(i) != c.n_classes - 1 else -1 for i in y_batch]
    acc_before = sum([1 if gt[i] == labels[i] else 0 for i in range(len(labels))]) / float(len(y_batch))
    print("*** ACC before clashes", acc_before)
    overall_acc_before += acc_before
    try:
        # if OoD are not of different shape
        # resolve clashes using confidence scores (logits values where argmax)
        classes = list(range(c.n_classes-1))
        classes.append(-1)
        for cl_ix in range(len(classes)):
            cl = classes[cl_ix]
            indices_with_label = np.where(np.array(labels) == cl)[0]
            if len(indices_with_label) > 1 and cl != -1:
                # keep the one with highest confidence (logit)
                all_logits = []
                for idx in indices_with_label:
                    all_logits.append(logits[idx][0][cl])
                highest_conf_pos_in_list = indices_with_label[np.argmax(all_logits)]
                idx_with_lower_conf = [i for i in indices_with_label if i != highest_conf_pos_in_list]
                for ix_lower in idx_with_lower_conf:
                    print("labels b4", labels)


                    # # assign it the next best logit
                    old_logits = logits[ix_lower][0]
                    # logits_without_highest = [old_logits[ix] for ix in range(len(old_logits)) if old_logits[ix] != old_logits[cl] and ix >= cl and ix < c.n_classes-1]
                    logits_without_highest = [old_logits[ix] for ix in range(len(old_logits)) if old_logits[ix] != old_logits[cl]]
                    new_max_ix = np.argmax(logits_without_highest)
                    new_label = np.where(old_logits == logits_without_highest[new_max_ix])[0][0]
                    # if new_label == c.n_classes:
                    #     new_label = -1
                    labels[ix_lower] = new_label
                    print("labels after", labels)
    except:
        pass

    acc_after = sum([1 if gt[i] == labels[i] else 0 for i in range(len(labels))]) / float(len(y_batch))
    overall_acc_after += acc_after
    print("*** ACC after clashes", acc_after)

    print("final labels", labels)


    # here comes your code to predict the labels of the images
    return labels, overall_acc_before, overall_acc_after


def evaluate(file_dir='output/', overall_acc_before=0, overall_acc_after=0):
    files = os.listdir(file_dir)
    # files.remove('.DS_Store')
    files.sort()
    images = []
    X_batch = []
    y_batch = []
    labels_in_folder = []
    for f in files:
        if f == '.DS_Store':
            continue
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        images.append(im)
        label = int(f.split('_')[-1].split('.')[0])
        labels_in_folder.append(label)

    t = get_t(images)
    img_size_dst = get_image_size(t)
    c = Conf(int(t), int(img_size_dst), is_image(images))
    labels_in_folder = [l if l != -1 else c.n_original_tiles for l in labels_in_folder]
    # labels_in_folder
        # change to n_original (e.g. for t=2 OoD tiles would get label 4 as labels 0,1,2,3 are original)
        # label = c.n_original_tiles

    print(labels_in_folder)
    # in case we're adding OoD null image
    if len(labels_in_folder) < c.n_tiles_per_sample:
        for k in range(c.n_tiles_per_sample-len(labels_in_folder)):
            labels_in_folder.append(-1)

    folder_labels = to_categorical(labels_in_folder, num_classes=c.n_classes)
    y_batch.append(folder_labels)
    y_batch = list(np.array(y_batch).reshape(c.n_tiles_per_sample, 1, c.n_classes))

    Y, overall_acc_before, overall_acc_after = predict(images, y_batch, overall_acc_before, overall_acc_after) # TODO
    # Y = predict(images) # TODO
    print(Y)
    return Y, overall_acc_before, overall_acc_after


all_accs = {}

n_to_check = 2

for t_ in [4,5, 2]:

    overall_acc_before = 0
    overall_acc_after = 0
    try:
        print("Starting {}". format(t_))
        files_dict = load_obj('train_test_val_dict_img_{}'.format(t_))
        files = files_dict['test']
        for f in files[:n_to_check]:
            # TODO: add another for loop to do the same also for augmented tiles, but make sure OoD is from a previous image, not from a previous augmentation
            _, overall_acc_before, overall_acc_after = evaluate('dataset_{}_isImg_True/'.format(t_)+f+'/', overall_acc_before, overall_acc_after)
    except:
        files_dict = load_obj('train_test_val_dict_doc_{}'.format(t_))
        files = files_dict['test']
        for f in files[:n_to_check]:
            # TODO: add another for loop to do the same also for augmented tiles, but make sure OoD is from a previous image, not from a previous augmentation
            _, overall_acc_before, overall_acc_after = evaluate('dataset_{}_isImg_False/'.format(t_) + f + '/', overall_acc_before, overall_acc_after)

    all_accs[t_] = (overall_acc_before / float(n_to_check), overall_acc_after / float(n_to_check))
print("ACCS B4 vs AFTER", all_accs)
    #     # filename = f.split('.')[0] + '_crw_' + str(c_w) + '_crh_' + str(c_h) + '_reshape_' + str(reshape)
    # evaluate('dataset_2_isImg_True/n01440764_172_crw_0_crh_0_reshape_False/')
    # read_test_images_docs('dataset_5_isImg_False/73_5_crw_0_crh_15_reshape_True/')
    # read_test_images_docs('dataset_5_isImg_True/n01440764_7267_crw_0_crh_45_reshape_False/')
    # calc_edge_similarity_score(read_test_images_docs('example/'))

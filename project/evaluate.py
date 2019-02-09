import os
import cv2
from keras.utils import to_categorical
from preprocessor import *
from resnet_adapted import *
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


def predict(images):
    # maybe_download_weights()  # no need - function called in  weights_download
    labels = []
    X_batch = []
    t = get_t(images)
    img_size_dst = get_image_size(t)
    c = Conf(int(t), int(img_size_dst), is_image(images))
    isImg = is_image(images)

    resnet = build_resnet(c.max_size, c.n_tiles_per_sample, c.n_classes, c.n_original_tiles, c.tiles_per_dim)

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

    # adding additional OoD images (null image) to reach t OoDs per sampled folder
    for n_null in range(add_n_null_images):
        resized_images.append(np.zeros((c.max_size, c.max_size)))
        original_images.append(np.zeros((c.max_size, c.max_size)))
    resized_images = add_similarity_channel(resized_images, original_images, c)

    X_batch.append(np.array(resized_images))  # a folder is one single sample
    X_batch = list(np.array(X_batch).reshape(c.n_tiles_per_sample, 1, c.max_size, c.max_size, 2))

    logits = resnet.predict_on_batch(X_batch)
    for l in logits:
        idx_max = l.argmax(axis=1)
        idx_max = int(idx_max)
        if idx_max == c.n_classes - 1:
            # OoD
            idx_max = -1
        labels.append(idx_max)

    # in case OoDs are of a different shape, we can easily label them:
    OoD_have_diff_shape = False
    for im_idx in range(len(images)):
        im = images[im_idx]
        if im_shapes[im.shape] <= t:
            labels[im_idx] = -1

    labels = labels[:len(images)]

    labels = [l if l != c.n_original_tiles else -1 for l in labels]
    if len(labels) == c.n_original_tiles and sum(labels) == -1*c.n_original_tiles:
        # choose some other random label as there are no OoDs
        labels = [0 for i in labels]

    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        print(f)  #TODO: remove
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)


    Y = predict(images)
    print(Y)  # TODO - remove!
    return Y


# # TODO: remove
# evaluate('dataset_2_isImg_True/n01440764_172_crw_0_crh_0_reshape_False/')

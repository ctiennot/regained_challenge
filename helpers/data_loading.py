import os
import pandas as pd
import numpy as np
from PIL import Image
import time


def load_faces(dir="data/extracted_faces/", verbose=True):
    """ Load faces images (in black and white)
    :param dir: "data/extracted_faces/" by default
    :return: X, ids
    """
    all_faces = {int(number[::-1][4:][::-1]): dir + number
                      for number in os.listdir(dir)
                      if number[::-1][:4] == 'gpj.'}

    n = len(all_faces)

    X = np.zeros((len(all_faces), 10**4))

    for i in range(n):
        if (i+1) % 1000==0 and verbose:
            print "{}/{} faces loaded".format(i+1, n)
        im = np.array(Image.open(all_faces[i+1])).astype(float)
        if len(im.shape)>2:
            X[i, :] = im.mean(axis=2).reshape((-1,))/255.
        else:
            X[i, :] = im.reshape((-1,)) / 255.

    ids = np.arange(1, n+1)

    return np.array(X, dtype=np.float32), ids


def load_labels(file="data/training_outputs.csv"):
    """
    :param file: "data/training_outputs.csv" by default
    :return: y, ids
    """
    # read .csv using pandas
    y = pd.read_csv(file, delimiter=";").set_index("ID")
    ids = np.array(y.index)
    y = y.as_matrix().reshape((-1))

    return y, ids


def load_meta_features(file="data/facial_features_train.csv", only_num=False):
    """
    :param file: "data/facial_features_train.csv" by default
    :param only_num: if True then discard non numerical features
    :return:
    """
    # read .csv using pandas
    X = pd.read_csv(file).set_index("ID")

    # store id's
    ids = np.array(X.index)

    # convert age strings to integers
    X["age"] = X["age"].apply(lambda x: 0 if x == "None" else int(x))
    # convert gender to integers
    X["gender"] = X["gender"].apply(
        lambda a: 2 * (a == 'male') + 1 * (a == 'female') + 0 * (
        a == 'None'))
    # convert left_eye to integers
    X["left_eye"] = X["left_eye"].apply(
        lambda a: 2 * (a == 'opened') + 1 * (a == 'closed') + 0 * (
        a == 'None'))
    X["right_eye"] = X["right_eye"].apply(
        lambda a: 2 * (a == 'opened') + 1 * (a == 'closed') + 0 * (
        a == 'None'))
    # convert confidence to float between 0.5 and 1 (0 if None)
    for col in ["confidence_gender", "confidence_left_eye",
                "confidence_right_eye"]:
        X[col] = X[col].apply(lambda x: 0 if x == "None" else float(x))

    if only_num:
        numeric_features = np.array(
            [type(value) is not str for value in X.as_matrix()[0, :]])
        numeric_features = list(X.columns[numeric_features])
        numeric_features.sort()
        X = X[numeric_features]

    return X, ids


def export_submission(scores_sub, name="submission"):
    scores_sub = np.array(scores_sub).reshape((-1,))
    if np.array(scores_sub).size != 3000:
        raise UserWarning("You must give 3000 scores")
    ids = np.arange(10**4 + 1, 13*10**3 +1)
    submission = np.array(np.vstack((ids, np.round(scores_sub))).T, dtype=int)
    submission = pd.DataFrame(submission, columns=["ID", "TARGET"])
    name_sub = "submissions/" + name + time.strftime("_%d%m_%H%M") + ".csv"
    submission.to_csv(name_sub, sep=";", index=False)


def global_views_batch(pict_indices, dir="data/global_views/", verbose=False, black_white=True):
    """ Load global views by batches
    :param pict_indices: list/array with indices of images to load in the batch
    :param dir: "data/extracted_faces/" by default
    :param black_white: if True converts RGB to black and white, if False then
    for b&w images we replicate three times the grey channel.
    :param verbose: display loading progression
    :return: X, ids
    """
    all_img = {int(number[::-1][4:][::-1]): dir + number
                      for number in os.listdir(dir)
                      if number[::-1][:4] == 'gpj.'}

    w, h = Image.open(all_img[1]).size

    # initialize empty bashes
    if black_white:
        X = np.zeros((len(pict_indices), w, h), dtype=np.float32)
    else:
        X = np.zeros((len(pict_indices), w, h, 3), dtype=np.float32)

    # fill them in
    for i, index in enumerate(pict_indices):
        if (i+1) % 100 == 0 and verbose:
            print "{}/{} global views loaded".format(i+1, len(pict_indices))
        # get image
        im = Image.open(all_img[index])
        # first case: we want to keep colors
        if not black_white:
            im = np.array(im).astype(np.float32)
            # for color img
            if len(im.shape)>2:
                X[i] = im
            # for B&W img we replicate three times
            else:
                X[i, :, :, 0] = im
                X[i, :, :, 1] = im
                X[i, :, :, 2] = im
        # second case: we want B&W
        else:
            # for color img
            if len(np.array(im).shape)>2:
                im = im.convert('L')  # convert in B&W first
                im = np.array(im).astype(np.float32)
                X[i, :, :] = im
            # for B&W img it's already good
            else:
                im = np.array(im).astype(np.float32)
                X[i, :, :] = im

    ids = np.array(pict_indices)
    return X, ids


def local_views_batch(pict_indices, dir="data/local_views/", verbose=False, black_white=True, seed=None):
    """ Load local views by batches
    :param pict_indices: list/array with indices of images to load in the batch
    it selects randomly 1 croped square among all those availables
    :param dir: "data/extracted_faces/" by default
    :param black_white: if True converts RGB to black and white, if False then
    for b&w images we replicate three times the grey channel.
    :param verbose: display loading progression
    :param seed: if one, set seed
    :return: X, ids
    """
    if seed is not None:
        np.random.seed(seed)

    def valid_file_name(name):
        """ Returns only filenames that represent images"""
        if not name.__contains__("jpg"):
            return None  # not an image (can be .gitignore for instance)
        # parse index and patch number
        index, patch_number = tuple(map(int, name.split(".")[0].split("_")))
        return index, patch_number

    all_img = {}

    for file_name in os.listdir(dir):
        tup_index_patch = valid_file_name(file_name)
        if tup_index_patch is None:
            pass
        else:
            index, patch_number = tup_index_patch
            if index in pict_indices:
                if index in all_img.keys():
                    all_img[index].append(patch_number)
                else:
                    all_img[index] = [patch_number]

    # we sort the patch number lists to avoid messing up with the seed
    for little_list in all_img.values():
        little_list.sort()

    # look at first img to know the dimensions
    img_ = all_img.items()[0]
    w, h = Image.open(dir + str(img_[0]) + '_' + str(img_[1][0]) + '.jpg').size

    # initialize empty bashes
    if black_white:
        X = np.zeros((len(pict_indices), w, h), dtype=np.float32)
    else:
        X = np.zeros((len(pict_indices), w, h, 3), dtype=np.float32)

    # fill them in
    for i, index in enumerate(pict_indices):
        if (i+1) % 100 == 0 and verbose:
            print "{}/{} local views loaded".format(i+1, len(pict_indices))
        # randomly sample a patch to load
        k = np.random.choice(all_img[index], 1, False)[0]
        name = dir + str(index) + '_' + str(k) + '.jpg'
        # get image
        im = Image.open(name)
        # first case: we want to keep colors
        if not black_white:
            im = np.array(im).astype(np.float32)
            # for color img
            if len(im.shape)>2:
                X[i] = im
            # for B&W img we replicate three times
            else:
                X[i, :, :, 0] = im
                X[i, :, :, 1] = im
                X[i, :, :, 2] = im
        # second case: we want B&W
        else:
            # for color img
            if len(np.array(im).shape)>2:
                im = im.convert('L')  # convert in B&W first
                im = np.array(im).astype(np.float32)
                X[i, :, :] = im
            # for B&W img it's already good
            else:
                im = np.array(im).astype(np.float32)
                X[i, :, :] = im

    ids = np.array(pict_indices)
    return X, ids


def faces_batch(pict_indices, dir="data/extracted_faces/", verbose=False, black_white=True, seed=None):
    """ Load faces by batches
    :param pict_indices: list/array with indices of images to load in the batch
    :param dir: "data/extracted_faces/" by default
    :param black_white: if True converts RGB to black and white, if False then
    for b&w images we replicate three times the grey channel.
    :param verbose: display loading progression
    :return: X, ids
    """
    return global_views_batch(
        pict_indices, dir=dir, verbose=verbose, black_white=black_white)
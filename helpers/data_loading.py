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


def load_meta_features(file="data/facial_features_train.csv"):
    """
    :param file: "data/facial_features_train.csv" by default
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
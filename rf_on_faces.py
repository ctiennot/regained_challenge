import os
import matplotlib.pyplot as plt
from scipy import misc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

all_faces = {int(number[::-1][4:][::-1]): "data/extracted_faces/" + number
                  for number in os.listdir("data/extracted_faces")
                  if number[::-1][:4] == 'gpj.'}

bw = misc.imread(all_faces[1]).mean(axis=2)/255.  # black and white normalized
bw = bw.reshape((-1,))  # 10k features

X = np.zeros((len(all_faces), 10**4))

for i in range(13000):
    if i % 1000==0:
        print i
    im = misc.imread(all_faces[i+1])
    if len(im.shape)>2:
        X[i,:] =  misc.imread(all_faces[i+1]).mean(axis=2).reshape((-1,))/255.
    else:
        X[i, :] = misc.imread(all_faces[i + 1]).reshape((-1,)) / 255.



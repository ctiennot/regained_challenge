import os
import matplotlib.pyplot as plt
from scipy import misc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from spearman import score_function
import time

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


# read .csv using pandas
y = pd.read_csv("data/training_outputs.csv", delimiter=";").set_index("ID")
y = y.as_matrix().reshape((-1,))

# store id's
X_ids = range(1, 13001)

# cut into test, train
X_train, X_test, y_train, y_test = train_test_split(X[0:10000], y, test_size=0.3, random_state=42)

# now we try the random forest
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=True)

start = time.time()
rf.fit(X_train, y_train)
print "Running time", round(time.time() - start), "s"

# MSE on train and test
print "train\t", round(rf.score(X_train, y_train), 2)
print "test\t", round(rf.score(X_test, y_test), 2)

preds = rf.predict(X_test)

plt.figure()
plt.plot(y_test, preds, "o")
plt.plot([0,25], [0, 25])
plt.show()

score_function(y_pred=np.round(preds), y_true=y_test)
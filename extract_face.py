import os
import matplotlib.pyplot as plt
from scipy import misc
import pandas as pd
import matplotlib.patches as patches

pictures_train = {int(number[::-1][4:][::-1]): "data/pictures_train/" + number
                  for number in os.listdir("data/pictures_train")}
pictures_test = {int(number[::-1][4:][::-1]): "data/pictures_test/" + number
                  for number in os.listdir("data/pictures_test")}

n = 10
f, axarr = plt.subplots(n, n)
for i in range(n**2):
    axarr[i // n, i % n].axis('off')
    axarr[i//n, i%n].imshow(misc.imread(pictures_train.values()[i]))
    axarr[i // n, i % n].add_patch(
        patches.Rectangle(
            (0.1, 0.1),
            0.5,
            0.5,
            fill=False  # remove background
        )
    )
plt.show()

###############################################################################
#  Find faces in pictures
###############################################################################

meta_features = pd.read_csv("data/facial_features_train.csv").set_index("ID")
faces_location = meta_features[["x0", "y0", 'width', 'height']]

for i in range(100, 200):
    pict_id = i
    test = faces_location.loc[pict_id]
    pict_test = misc.imread(pictures_train[pict_id])
    pict_height, pict_width = pict_test.shape[0:2]

    # find the face
    face_x, face_y = int(test["x0"] * pict_width), int(test["y0"] * pict_height)
    face_w, face_h = int(test["width"] * pict_width), int(test["height"] * pict_height)

    # cropped image
    cropped = pict_test[face_y:(face_y+face_h), face_x:(face_x+face_w)]
    plt.figure()
    plt.imshow(cropped)
    plt.show()



    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.imshow(pict_test)

    ax2.add_patch(
        patches.Rectangle(
            (face_x, face_y),
            face_w, face_h,
            0.5,
            fill=False      # remove background
        )
    )
    plt.show()


###############################################################################
#  Get sizes of faces (rectangle or squares???)
###############################################################################
face_ratio = []
extracted_faces = []

for i in range(1, 200):
    pict_id = i
    test = faces_location.loc[pict_id]
    pict_test = misc.imread(pictures_train[pict_id])
    pict_height, pict_width = pict_test.shape[0:2]

    # find the face
    face_x, face_y = int(test["x0"] * pict_width), int(test["y0"] * pict_height)
    face_w, face_h = int(test["width"] * pict_width), int(test["height"] * pict_height)

    # cropped image
    cropped = pict_test[face_y:(face_y+face_h), face_x:(face_x+face_w)]
    cropped = misc.imresize(cropped, (100, 100))

    misc.imsave("data/extracted_faces/{}.jpg".format(pict_id), cropped)
    extracted_faces.append(cropped)

    face_ratio.append(face_h*1./face_w)

plt.plot(face_ratio)


plt.imshow(misc.imresize(cropped, (100, 100)))



###############################################################################
# Save extracted faces
###############################################################################
pictures = pictures_test
meta_features = pd.read_csv("data/facial_features_test.csv").set_index("ID")
faces_location = meta_features[["x0", "y0", 'width', 'height']]

for it, i in enumerate(pictures.keys()):
    if it % 100 == 0:
        print "{}/{}".format(it, len(pictures))
    pict_id = i
    test = faces_location.loc[pict_id]
    pict_test = misc.imread(pictures[pict_id])
    pict_height, pict_width = pict_test.shape[0:2]

    # find the face
    face_x, face_y = int(test["x0"] * pict_width), int(test["y0"] * pict_height)
    face_w, face_h = int(test["width"] * pict_width), int(test["height"] * pict_height)

    # cropped image
    cropped = pict_test[face_y:(face_y+face_h), face_x:(face_x+face_w)]
    cropped = misc.imresize(cropped, (100, 100))

    misc.imsave("data/extracted_faces/{}.jpg".format(pict_id), cropped)
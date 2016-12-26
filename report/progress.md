*This is a simple file to keep track of my progress, but the proper report will be written in latex.*

-------------------

**Monday, 19. December 2016 01:08pm **

So far I only used the provided features, excluding those that could not be directly converted to be nurmeric. I tried a simple random forest on that (0.43) and boosting (0.506).


**Samedi, 24. décembre 2016 04:22 **

Began to look at the pictures and the detected faces. Some pictures are in black and white (no 3rd dimension in numpy) and some other are in colors. Extract faces with the given position, width and height, resize them to be 100*100 and export them as images to the extracted_faces folder.


**dimanche, 25. décembre 2016 12:16 **

Random forest on face pixels (100*100) => 0.45

**lundi, 26. décembre 2016 12:01 **

Trained a CNN from scratch (reusing the tutorial from tensorflow deep mnist) on the extracted faces and reached about 0.50 score (about the same as with boosting) without much care about the learning rate.

=> next idea: push the CNN, fine tuning and/or get an embedding to put the features in the boosting along the meta features. 
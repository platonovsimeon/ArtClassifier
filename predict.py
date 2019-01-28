from keras.models import load_model
from PIL import Image as PilImage
from keras.preprocessing import image
import numpy
import os
import sys
import tensorflow as tf

classifier = load_model("my_model.h5")
img_path = sys.argv[1]

img = PilImage.open(img_path)
img = img.resize((64,64), PilImage.ANTIALIAS)
tmp_path= "tmp3245675289499032485.jpg"
img.save(tmp_path)

prediction_image = image.load_img(tmp_path,(64,64))
prediction_image = image.img_to_array(prediction_image)
prediction_image = numpy.expand_dims(prediction_image,axis=0)
os.remove("tmp3245675289499032485.jpg")
prediction = classifier.predict(prediction_image)[0]
indx = tf.arg_max(prediction,0)

with tf.Session() as sess:
    print(prediction)
    indx = sess.run(indx)
    if indx == 0:
        print("Drawing")
    if indx == 1:
        print("Iconography")
    if indx == 2:
        print("Painting")
    if indx==3:
        print("Sculpture")

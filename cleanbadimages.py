#This code was used to delete images that cause error during the training
import os
import numpy as np
from PIL import Image

def clean(folder):
    for file in os.listdir(folder):
        try:
            image = np.asarray(Image.open(folder+file))
            if len(image.shape)!=3:
                os.remove(folder+file)
            else:
                if image.shape[2]!=3:
                    os.remove(folder+file)

        except OSError:
            os.remove(folder+file)

clean("data/train/drawings/")
clean("data/train/iconography/")
clean("data/train/painting/")
clean("data/train/sculpture/")
clean("data/validation/drawings/")
clean("data/validation/iconography/")
clean("data/validation/painting/")
clean("data/validation/sculpture/")

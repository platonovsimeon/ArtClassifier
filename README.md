# ArtClassifier

# What is this project about?
This project contains a neural network that aims to classify four types of art: painting, iconography, sculpture, and drawing. Instead of a traditional CNN, this is a residual convolutional neural network.

# How to use this project?
The root folder of this project contains five files + readme. It also contains a folder which contains the training data. 
1. cleanbadimages.py was used to delete images from the dataset which caused errors during the training. You do not need to run it.
2. index.jpg is an image that you can use to test this neural network.
3. my_model.zip contains a trained network.
4. predict.py can make a prediction on a new image. Just type following to your command line: python predict.py index.jpg. You need to unzip my_model.zip before using predict.py.
5. train.py can be used to train a new network on the data.

# Note on the results
This model seems to be quite good at classifying sculptures and icons. However, when you give it a painting or a drawing it performs poorly. Committing more data and improving the network architecture is encouraged.


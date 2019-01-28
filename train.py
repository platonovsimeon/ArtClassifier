from keras.layers import Input,Conv2D, Dense, Flatten, MaxPooling2D,Add,Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

input = Input(shape=(64,64,3))
layer1 = Conv2D(256,3,padding="same",activation="relu")(input)
layer1 = MaxPooling2D()(layer1)

res = layer1
layer2 = Conv2D(256,3,padding="same",activation="relu")(layer1)
layer2 = Add()([res,layer2])
layer2 = MaxPooling2D()(layer2)

res = layer2
layer3 = Conv2D(256,3,padding="same",activation="relu")(layer2)
layer3 = Add()([res,layer3])
layer3 = MaxPooling2D()(layer3)

res = layer3
layer4 = Conv2D(256,3,padding="same",activation="relu")(layer3)
layer4 = Add()([res,layer4])

layer5 = Flatten()(layer4)
layer5 = Dense(512,activation="relu")(layer5)
layer5 = Dropout(rate=0.25)(layer5)

layer6 = Dense(4,activation="softmax")(layer5)
out = layer6

model = Model(inputs=input,outputs=out)
model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',target_size=(64,64),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'dataset/validation',
        target_size=(64,64),
        batch_size=20,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=240,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=22)

model.save('my_model.h5')

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, merge
from keras.models import Model
from keras import backend as K
K.set_image_dim_ordering('th')

# this is the model we use now. basically unet, with an extra block pair, and batch norm
# note: as a result of using batch norm, the model's weights is locked to input_shape
# TODO should probably make a new net without batch norm....
def get_unet(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(256, (3, 3), activation = "relu", padding = "same")(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation = "relu", padding = "same")(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(bn5)
    bn5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)

    conv5_2 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(pool5)
    bn5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(bn5_2)
    bn5_2 = BatchNormalization()(conv5_2)

    up5_2 = Concatenate(axis = 1)([UpSampling2D(size = (2, 2))(bn5_2), bn5])
    conv6_2 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(up5_2)
    bn6_2 = BatchNormalization()(conv6_2)
    conv6_2 = Conv2D(512, (3, 3), activation = "relu", padding = "same")(bn6_2)
    bn6_2 = BatchNormalization()(conv6_2)

    up6 = Concatenate(axis = 1)([UpSampling2D(size = (2, 2))(bn6_2), bn4])
    conv6 = Conv2D(256, (3, 3), activation = "relu", padding = "same")(up6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation = "relu", padding = "same")(bn6)
    bn6 = BatchNormalization()(conv6)

    up7 = Concatenate(axis = 1)([UpSampling2D(size = (2, 2))(bn6), bn3])
    conv7 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(up7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation = "relu", padding = "same")(bn7)
    bn7 = BatchNormalization()(conv7)

    up8 = Concatenate(axis = 1)([UpSampling2D(size = (2, 2))(bn7), bn2])
    conv8 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(up8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation = "relu", padding = "same")(bn8)
    bn8 = BatchNormalization()(conv8)

    up9 = Concatenate(axis = 1)([UpSampling2D(size = (2, 2))(bn8), bn1])
    conv9 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(up9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(bn9)
    bn9 = BatchNormalization()(conv9)

    conv10 = Conv2D(int(input_shape[0]/2), (1, 1), activation = "sigmoid")(bn9)

    model = Model(inputs = inputs, outputs = conv10)

    return model
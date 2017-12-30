from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Cropping3D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.models import Model

K.set_image_dim_ordering('th')

# For understanding the architecture itself, I recommend checking the following article
# Dolz, J. et al. 3D fully convolutional networks for subcortical segmentation in MRI :
# A large-scale study. Neuroimage, 2017.
def generate_model(num_classes, num_channel=1, input_size=(27, 27, 27), output_size=(9, 9, 9)) :
    init_input = Input((num_channel,) + input_size)

    x = Conv3D(25, kernel_size=(3, 3, 3))(init_input)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)

    y = Conv3D(50, kernel_size=(3, 3, 3))(x)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)

    z = Conv3D(75, kernel_size=(3, 3, 3))(y)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)

    x_crop = Cropping3D(cropping=((6, 6), (6, 6), (6, 6)))(x)
    y_crop = Cropping3D(cropping=((3, 3), (3, 3), (3, 3)))(y)

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = Conv3D(400, kernel_size=(1, 1, 1))(concat)
    fc = PReLU()(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)

    pred = Conv3D(num_classes, kernel_size=(1, 1, 1))(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, output_size[0]*output_size[1]*output_size[2]))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)

    model = Model(inputs=init_input, outputs=pred)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    return model
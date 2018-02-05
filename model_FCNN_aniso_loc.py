from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Cropping3D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.models import Model

K.set_image_dim_ordering('th')

# Ref:
#      Dolz, J. et al. 3D fully convolutional networks for subcortical segmentation in MRI :
#      A large-scale study. Neuroimage, 2017.
def generate_model(num_classes, num_channel=1, input_size=(27, 27, 9), output_size=(9, 9, 3), num_dim_loc=3) :
    image_input = Input((num_channel,) + input_size)
    loc_input = Input((num_dim_loc,) + output_size)

    x = Conv3D(25, kernel_size=(7, 7, 3))(image_input)
    x = PReLU()(x)
    x = BatchNormalization(axis=1)(x)
    
    y = Conv3D(50, kernel_size=(7, 7, 3))(x)
    y = PReLU()(y)
    y = BatchNormalization(axis=1)(y)

    z = Conv3D(75, kernel_size=(7, 7, 3))(y)
    z = PReLU()(z)
    z = BatchNormalization(axis=1)(z)
    
    #l = BatchNormalization(axis=1)(loc_input)

    x_crop = Cropping3D(cropping=((6, 6), (6, 6), (2, 2)))(x)
    y_crop = Cropping3D(cropping=((3, 3), (3, 3), (1, 1)))(y)

    #concat = concatenate([x_crop, y_crop, z], axis=1)
    concat = concatenate([x_crop, y_crop, z, loc_input], axis=1)

    fc = Conv3D(400, kernel_size=(1, 1, 1))(concat)
    fc = PReLU()(fc)
    fc = BatchNormalization(axis=1)(fc)
    fc = Dropout(0.2)(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = BatchNormalization(axis=1)(fc)
    fc = Dropout(0.2)(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = BatchNormalization(axis=1)(fc)
    fc = Dropout(0.2)(fc)
    
    pred = Conv3D(num_classes, kernel_size=(1, 1, 1))(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, output_size[0]*output_size[1]*output_size[2]))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)

    model = Model(inputs=[image_input, loc_input], outputs=pred)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    return model

from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Flatten, Dense
from tensorflow.keras.layers import AveragePooling2D, add, Input, ZeroPadding2D, MaxPool2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow import nn as tfn
import tensorflow.keras.backend as K

class ResNet:
    @staticmethod
    def residualModel(x, k, stride, chanDim, dim_reduce=False, eps=2e-5, m=0.9,
                      reg_val= 0.0001):
        short_branch= x

        bn1= BatchNormalization(axis=chanDim,epsilon=eps,momentum=m)(x)
        act1= Activation(tfn.relu)(bn1)
        conv1= Conv2D(filters=(k//4), kernel_size=(1,1), kernel_regularizer=l2(reg_val),
                      use_bias=False)(act1)

        bn2= BatchNormalization(axis=chanDim,epsilon=eps,momentum=m)(conv1)
        act2 = Activation(tfn.relu)(bn2)
        conv2 = Conv2D(filters=(k // 4), kernel_size=(3, 3), strides=stride, padding='same',
                       kernel_regularizer=l2(reg_val),
                       use_bias=False)(act2)

        bn3 = BatchNormalization(axis=chanDim, epsilon=eps, momentum=m)(conv2)
        act3 = Activation(tfn.relu)(bn3)
        conv3 = Conv2D(filters=k, kernel_size=(1, 1), kernel_regularizer=l2(reg_val),
                       use_bias=False)(act3)

        if dim_reduce:
            short_branch= Conv2D(filters=k, kernel_size=(1,1), strides=stride,
                                 kernel_regularizer=l2(reg_val), use_bias=False)(act1)

        x= add([conv3, short_branch])

        return x

    @staticmethod
    def build(width, height, depth, classes, num_layers, num_filters, eps=2e-5, m=0.9, reg_val=0.0001):
        inputShape= (height, width, depth)
        chanDim=-1

        if K.image_data_format()=='channels_first':
            inputShape= (depth, height, width)
            chanDim= 1

        inputs= Input(shape=inputShape)
        initial_bn= BatchNormalization(axis=chanDim, epsilon=eps, momentum=m)(inputs)
        x= Conv2D(filters=num_filters[0], kernel_size=(5,5), padding='same',
                  use_bias=False, kernel_regularizer=l2(reg_val))(initial_bn)
        x= BatchNormalization(axis=chanDim, epsilon=eps, momentum=m)(x)
        x= Activation(tfn.relu)(x)
        # Zero padding is added for proper downsampling since maxpooling uses a (3,3) instead of (2,2)
        x= ZeroPadding2D((1,1))(x)
        x= MaxPool2D(pool_size=(3,3), strides=2)(x)

        for i in range(0, len(num_layers)):
            if i==0:
                stride=1
            else:
                stride=2

            # Only first set of residuals don't downsample everyone else does
            x= ResNet.residualModel(x, k=num_filters[i+1], stride=stride, chanDim=chanDim,
                                    dim_reduce=True, eps=eps, m=m, reg_val=reg_val)

            for j in range(0, num_layers[i]-1):
                x= ResNet.residualModel(x, k=num_filters[i+1], stride=1,
                                        chanDim=chanDim, eps=eps, m=m, reg_val=reg_val)

        x= BatchNormalization(axis=chanDim, momentum=m, epsilon=eps)(x)
        x= Activation(tfn.relu)(x)
        x= AveragePooling2D(pool_size=(8,8))(x)
        x= Flatten()(x)
        x= Dense(classes, kernel_regularizer=l2(reg_val))(x)
        x= Activation(tfn.softmax)(x)

        model= Model(inputs, x, name="resnet")

        return model
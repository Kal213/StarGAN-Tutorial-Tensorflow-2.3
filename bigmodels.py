#SAME IDEA AS models.py BUT WITH MORE DOWNSAMPLING AND UPSAMPLING FOR HIGHER RESOLUTION.
#VERY UNSTABLE, AND EASY TO COLLAPSE.  I'd recommend 16 cha, and no higher than 256x256 images.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

#Code for adaptive instance normalization itself.  Normalizes the data around a learned
#mean and standard deviation from the label input.
def AdaIn(x):
    contentOut = x[0]
    styledev = x[1]
    stylemean = x[2]
    mean = K.mean(contentOut, axis =[1,2], keepdims = True)
    std = K.std(contentOut, axis=[1,2], keepdims = True) + 1e-7
    y = (contentOut - mean) / std #normalize gamma and beta.

    pool_shape = [-1,1,1,y.shape[-1]]
    g = K.reshape(styledev,pool_shape)
    b = K.reshape(stylemean, pool_shape)

    return y * g + b

#standard convolutional downsampling layer for the generator.
def genDownBlock(inp, channels):
    x = Conv2D(channels, 3, padding="same", kernel_initializer='glorot_uniform')(inp)
    x = LeakyReLU(0.2)(x)
    x = AveragePooling2D()(x)
    x = Conv2D(channels*2, 3, padding="same", kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)
    return x

#standard convolutional upsampling layer for the generator.
def genUpBlock(inp, channels):
    x = Conv2D(channels, 3, padding="same", kernel_initializer='glorot_uniform')(inp)
    x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x)
    x = Conv2D(channels//2, 3, padding="same", kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)
    return x

#Code block for the adaptive instance normalization method of incorporating the label.
#Uses residual connections in order to maintain original structure.
def adaBlock(inp, inp_target, channels):
    y = Conv2D(channels, 1, padding="same", kernel_initializer='glorot_uniform')(inp)

    x = Conv2D(channels, 3, padding="same", kernel_initializer='glorot_uniform')(inp)
    g = Dense(channels, bias_initializer='ones')(inp_target)
    b = Dense(channels)(inp_target)
    x = Lambda(AdaIn)([x, g, b])
    x = LeakyReLU(0.2)(x)

    x = Add()([x,y])

    return x

def makeGen(cha, NUMLABELS):

    inp = Input((256,256,3))

    inp_target = Input([NUMLABELS])

    #Here's a method of tiling the input.  If using this, remove AdaIN and replace with residual blocks.
    # y = Reshape((1,1,inp_target.shape[-1]))(inp_target)
    # tile = Lambda(lambda a: tf.tile(a[0],[1,a[1],a[2],1]))([y,inp.shape[1],inp.shape[2]])
    # x = Concatenate(axis=-1)([inp,tile])

    #Different names to allow for skip connections later in the model.
    #Helps to maintain structure.
    x1 = genDownBlock(inp,2*cha)
    x2 = genDownBlock(x1,4*cha)
    x3 = genDownBlock(x2,8*cha)
    x4 = genDownBlock(x3,16*cha)
    x5 = genDownBlock(x4,32*cha)

    x = adaBlock(x5,inp_target,cha*64)
    x = adaBlock(x,inp_target,cha*64)

    x = Concatenate(axis=-1)([x,x5]) #An example of a skip connection: Earlier data is concatenated channel wise to the current data.
    x = genUpBlock(x, cha * 64)

    x = Concatenate(axis=-1)([x,x4])
    x = genUpBlock(x, cha * 32)

    x = Concatenate(axis=-1)([x,x3])
    x = genUpBlock(x, cha * 16)

    x = Concatenate(axis=-1)([x,x2])
    x = genUpBlock(x, cha * 8)

    x = Concatenate(axis=-1)([x,x1])
    x = genUpBlock(x, cha * 4)

    x = Concatenate(axis=-1)([x,inp])
    x = Conv2D(cha, 3, padding="same", kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(3, 3, padding="same", kernel_initializer='glorot_uniform')(x)

    model = tf.keras.Model(inputs=[inp,inp_target], outputs=x)

    return model

#Standard residual downsampling layer for the discriminator.
def discDownBlock(inp, channels):
    x = Conv2D(channels, 3, padding="same", kernel_initializer='glorot_uniform')(inp)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(channels, 3, padding="same", kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(0.2)(x)

    res = Conv2D(channels, 1, padding="same", kernel_initializer='glorot_uniform')(inp)
    x = add([res,x])

    x = AveragePooling2D()(x)

    return x

#The discriminator is a multi-task discriminator, meaning that it gives outputs
#for each label regarding how real or fake it appears according to that class.
def makeDisc(cha, NUMLABELS):
    inp = Input((256,256,3))

    x = discDownBlock(inp,2*cha)
    x = discDownBlock(x,4*cha)
    x = discDownBlock(x,8*cha)
    x = discDownBlock(x,16*cha)
    x = discDownBlock(x,32*cha)
    x = GlobalAveragePooling2D()(x)

    x = Dense(cha*16)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(cha*16)(x)
    x = LeakyReLU(0.2)(x)
    x1 = Dense(NUMLABELS)(x) #Normal output

    model = tf.keras.Model(inputs=inp, outputs=x1)
    return model

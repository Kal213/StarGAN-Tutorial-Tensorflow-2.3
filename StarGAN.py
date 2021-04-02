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
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from PIL import Image
from math import log2
import random
from datagen import dataGenerator, printProgressBar
from models import makeGen, makeDisc
import bigmodels

class GAN:
    def __init__(self, data, test_data, image_size, model_name = "StarGAN", channels=16, size="normal", verbose=False, batch_size = 6, learning_rate = 0.0001):
        #data: A list of folder names inside of the data folder to generate data from for training.
        #test_data: A list of folder names inside of the data folder to generate data for testing.  Can be the same as data.
        #image_size: A tuple depicting the size of the desired size of the images for training.  The data generator will resize the images to this size.
        #model_name: A name for the model.  Used for the folder in which results and checkpoints are saved.
        #channels: The number of channels to be used at each step in the model before multiplication.  Recommended 16.
        #size: Can be "normal" or "large".  Determines which model to use.  Large is less stable, and more prone to collapse.
        #verbose: Whether or not the data generators will create output showing their status while generating the data.
        #batch_Size: The batch size for the model.
        #learning_rate: The learning rate for the model.  The discriminator's will be double this value.


        self.MODELNAME = model_name
        self.CKPT = os.path.dirname(os.path.realpath(__file__)) + "\\" + self.MODELNAME + "\\checkpoints\\"
        self.imagedir = os.path.dirname(os.path.realpath(__file__)) + "\\" + self.MODELNAME
        self.verbose = verbose
        self.size = size

        #Try making each directory, if it fails it generally means the folder already exists, so continue regardless.
        try:
            os.makedirs(self.imagedir)
        except OSError as error:
            pass
        try:
            os.makedirs(self.CKPT)
        except OSError as error:
            pass

        #Create both the training and testing datagenerators using a list of strings
        #containing the folder names inside of the data folder.
        #The first string will have the first label, and so on.
        self.datagens = []
        for item in data:
            self.datagens.append(dataGenerator(item, image_size, verbose = self.verbose, resize=True))
        self.testData= []
        for item in test_data:
            self.testData.append(dataGenerator(item, image_size, verbose = self.verbose, resize=True))

        #Determine the number of labels in the model.
        self.NUMLABELS = len(self.datagens)

        #Make the generator and discriminator as specified in models.py either normal or large sized.
        self.cha = channels
        self.BATCH_SIZE = batch_size
        if(self.size == "normal"):
            self.gen = makeGen(self.cha, self.NUMLABELS)
            self.disc = makeDisc(self.cha, self.NUMLABELS)
        elif(self.size == "large"):
            self.gen = bigmodels.makeGen(self.cha, self.NUMLABELS)
            self.disc = bigmodels.makeDisc(self.cha, self.NUMLABELS)


        #Setup the optimizers
        self.discOpt = tf.keras.optimizers.Adam(learning_rate=learning_rate*2, beta_1=0.0, beta_2=0.99)#
        self.genOpt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99)#

    @tf.function
    def trainStep(self, images, labels):
        #function to train the model.

        def grad_loss(samples, output, k=1.0):
        #An implementation of a two-sided local gradient penalty.
        #Helps to smooth out gradients in the discriminator.
        #Not strictly necessary, used to improve stability of discirminator.

            init_grad = tf.gradients(output, samples)[0]
            squared_grad = tf.square(init_grad)
            sum_grad = tf.sqrt(K.sum(squared_grad, axis=[1,2,3]))
            penalty = tf.maximum(sum_grad-k, tf.keras.backend.zeros_like(sum_grad))

            return tf.reduce_mean(penalty)

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            #Running data through models
            generatedImage = self.gen([images,labels[1]],training=True)
            restoredImage = self.gen([generatedImage,labels[0]], training=True)
            genfakeOut = K.sum(self.disc([generatedImage],training=True) * labels[1], axis=1) #Multiply by label due to multi-task discriminator.
            discRealOut = K.sum(self.disc([images],training=True) * labels[0], axis=1) #Multiply by label due to multi-task discriminator.

            #Loss functions
            cycleLoss = K.mean(K.abs(images - restoredImage)) * 5
            genLoss = K.mean(genfakeOut) + cycleLoss #Due to multi-task discriminator, label comparison and real/fake are done in one with genfakeout.
            discLoss = K.mean(K.relu(1.0 - genfakeOut) + K.relu(1.0 + discRealOut)) + K.mean(grad_loss(images, discRealOut)) * 10 #Hinge loss plust gradient penalty.

        #Calculate and apply gradients.
        genGradients = genTape.gradient(genLoss,self.gen.trainable_variables)
        discGradients = discTape.gradient(discLoss,self.disc.trainable_variables)

        self.genOpt.apply_gradients(zip(genGradients,self.gen.trainable_variables))
        self.discOpt.apply_gradients(zip(discGradients,self.disc.trainable_variables))
        return (genLoss, discLoss)

    def labelMaker(self, index, maxSize=None, batch=None):
        #Creates a one hot vector for the label of the image.
        #Index: the index for where the value will be one.
        #maxSize: typically the number of labels.  How long to make the vector.
        #batch: the batch size, or how many labels to produce.

        if maxSize == None:
            maxSize = self.NUMLABELS
        if batch == None:
            batch = self.BATCH_SIZE
        labels = np.ones([batch]) * index
        return to_categorical(labels, num_classes = maxSize)

    def train(self, steps = 100000, curStep = 1):
        #The train function repeats the training step to train the model.
        #steps: The number of steps to train.
        #curStep: The step to begin training on.  (e.g. In case you load weights from 50000 steps and want to retrain from 50000 steps onward.)

        #Setup some variables to compute train time and store loss values.
        genLoss = 0
        discLoss = 0
        trainTime = time.time()
        start = time.time()

        for step in range(curStep,steps+1):

            #Randomly select a source and target label and batch.
            randInt = random.randint(0, self.NUMLABELS-1)
            batch = self.datagens[randInt].get_batch(self.BATCH_SIZE)
            smalllabelsReal = self.labelMaker(randInt)

            #Selects a class to convert to.
            randInt = (random.randint(1,self.NUMLABELS-1)+randInt) % self.NUMLABELS
            smalllabelsNew = self.labelMaker(randInt)

            labels = [smalllabelsReal,smalllabelsNew]

            stepGenLoss, stepDiscLoss = self.trainStep(batch, labels)


            #Print the progress bar so that you may see how far along the model is.
            printProgressBar(step % 1000, 999, decimals=2)

            #Save variables to compute the average.
            genLoss += stepGenLoss
            discLoss += stepDiscLoss

            if (step) % 1000 == 0:
            #At every 1000 steps, generate an image containing all possible conversions, and show the average loss values for the previous 1000 steps.
                self.makeImages(step)
                print("At step {}.  Time for 1000 steps was {} seconds".format(step,time.time()-start))
                print("Average generator loss: {}, Average discriminator loss: {}".format((genLoss / 1000.0),(discLoss / 1000.0)))
                genLoss = 0
                discLoss = 0
                start = time.time()
            if (step) % 10000 == 0:
            #At every 10000 steps, save the weights of the generator and discriminator so they may be loaded.
                self.gen.save_weights(self.CKPT + f"{step}GEN.ckpt")
                self.disc.save_weights(self.CKPT + f"{step}DISC.ckpt")
                print("Gan saved!")

        #At the end of training, show the total amount of time it took.
        print(f"Time for training was {(time.time() - trainTime) / 60.0} minutes")
        return

    def makeImages(self, step, numEx = 5):
        #A function to create an array of images.  The first row is real, the second row is converted to the first class, the third row is converted to the second class, and so on.
        #Step: A only used in naming.  Typically used to show what step the image was generated on.

        imageRows = []
        #For each class, translate to each other class.
        #Original images will be on the top row in order of label.
        #Every row beneath will be a different label.
        for i in range(self.NUMLABELS):
            batch = self.testData[i].get_batch(numEx)

            #Place all of the original images on one row.
            rowstart = batch[0]
            for k in range(1,numEx):
                rowstart = np.append(rowstart, batch[k], 1)

            for j in range(self.NUMLABELS):
                results = self.gen([batch, self.labelMaker(j, self.NUMLABELS, numEx)], training=False)
                if i == j: #Don't convert to your own class!  Instead show a black box.
                    results = np.zeros_like(results)
                rowAdd = results[0]
                for k in range(1,numEx):
                    rowAdd = np.append(rowAdd, results[k], 1)
                rowstart = np.append(rowstart, rowAdd, 0)
            imageRows.append(rowstart)

        output = imageRows[0]
        for i in range(1,len(imageRows)):
            output = np.append(output, imageRows[i], 1) #All originals will be on

        output = np.clip(np.squeeze(output), 0.0, 1.0)
        plt.imsave(self.imagedir + f"\\{step}.png", output)

    def loadSave(self, step):
        #A function to load the weights of a trained model.  Must be the same size and channels as initialized model.
        #Step: What step to laod from.
        self.gen.load_weights(self.CKPT + f"{step}GEN.ckpt")
        self.disc.load_weights(self.CKPT + f"{step}DISC.ckpt")

    def translate(self, image, target):
        #Converts a single image to the target class.  Returns the translated image.
        #image: an array of (imageSize1,imageSize2,channels).
        #target: an index for what class to convert to (starting at 0)
        image = np.expand_dims(image, 0)
        label = self.labelMaker(target, batch=1)
        return np.squeeze(self.gen([image, label], training=False), axis=0)

#An example of how you could run the model using class folders "/data/classA_folder/", etc image size 256, model name "StarGAN", channel coefficient of 16, and normal size.
#if __name__ == "__main__":
#    data = ["classA_folder", "classB_folder", "classC_folder"] #In this case, class A has an index of 0, B 1, C 2.
#    testdata = ["classA_testfolder", "classB_testfolder", "classC_testfolder"]
#    starGAN = GAN(data, testdata, 256, "StarGAN", 16, "normal")
#    starGAN.makeImages(-999)
#    starGAN.train(200000)
#    exit()

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import warnings

import cv2

import keras
import keras.backend as K

from keras import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from keras.layers import BatchNormalization, Activation, Conv2D
from keras.layers import GlobalAveragePooling2D, Lambda
from keras.optimizers import Adam, RMSprop

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')

print('Keras version:', keras.__version__)

warnings.simplefilter('default')


# #### Set a few global parameters and directories

# In[3]:


'''
train_path = './train/'
non_landmark_train_path = './distractors/*/'
dev_path = './dev/'
non_landmark_dev_path = './distractors-dev/'
test_path = './test-highres/'

n_cat = 14942

batch_size = 48
batch_size_predict = 128
input_shape = (299,299)'''


# In[4]:


train_path = './train/'
#non_landmark_train_path = './distractors/*/'
#test_path = './test-highres/'
train_info_full = pd.read_csv('train.csv',index_col='id' )
train_image_files = glob.glob(train_path+'*.jpg')
#train_image_files = train_image_files[:100] #should remove this 
train_image_ids = [image_file.replace('.jpg', '').replace(train_path, '') for image_file in train_image_files]
train_info = train_info_full.loc[train_image_ids]
n_cat_train=train_info['landmark_id'].nunique()
train_info['filename'] = pd.Series(train_image_files, index=train_image_ids)


batch_size = 48
batch_size_predict = 128
input_shape = (299,299)
train_info.head()


# In[ ]:


len(train_image_files)


# #### Data preparation
# 
# Most of the code lines deal with missing images and the fact that I had started with low resolution images and that the high resolution image collection had different missing images compared to the low resolution collection.
# 
# Basically, the following lines load the dataframes provided by kaggle, remove all missing images and add a field `filename` with a path to the downloaded jpg file.
# 
# There are 5 dataframes:
# * train_info: train, landmark images
# * nlm_df: train, non-landmark images
# * dev_info: dev, landmark images
# * nlm_dev_df: dev, non-landmark images
# * test_info: test images

# In[ ]:


"""non_landmark_image_files=glob.glob(non_landmark_train_path + '*.jp*g')
nlm_df=pd.DataFrame({'filename': non_landmark_image_files})
nlm_df['landmark_id']=-1
print(len(nlm_df))
nlm_df.head()"""


# In[ ]:


n_cat_train=train_info['landmark_id'].nunique()
n_cat = n_cat_train
print(n_cat_train)


# In[ ]:


#print(train_info['url'])


# In[ ]:


label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=True, n_values=n_cat_train)

train_info['label'] = label_encoder.fit_transform(train_info['landmark_id'].values)
train_info['one_hot'] = one_hot_encoder.fit_transform(train_info['label'].values.reshape(-1, 1))


# In[ ]:


def load_images(filenames):
#    imgs = None
    imgs = []
    for i,filenames in enumerate(filenames):
        fname = filenames
        try:
            img = cv2.cvtColor(
                  cv2.resize(cv2.imread(fname),(input_shape)),
                  cv2.COLOR_BGR2RGB)
        except:
            warnings.warn('Warning: could not read image: '+ fname +
                          '. Use black img instead.')
            img = np.zeros((input_shape[0], input_shape[1], 3))
        imgs.append(img)
        if i %10000 == 0:
            print(len(imgs))
    np_img = np.asarray(imgs)
    print(np_img.shape)
    
    return np_img


# In[ ]:


from sklearn.model_selection import train_test_split

X = load_images(train_info['filename'])
y = train_info['landmark_id']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)


# In[ ]:


datagen = ImageDataGenerator(
            rotation_range=4.,rescale=1./255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.5,
            channel_shift_range=25,
            horizontal_flip=True,
            fill_mode='nearest')



datagen.fit(X_train)

train_generator = datagen.flow(X_train, y=y_train, batch_size=128)
#validation_generator = datagen.flow(x_val, y=y_val, batch_size=32)


# #### The NN model
# 
# Let's build the actual model!

# In[ ]:


K.clear_session()


# In[ ]:


"""from keras.models import load_model

# Returns a compiled model identical to the previous one"""



x_model = Xception(input_shape=list(input_shape) + [3], 
                   weights=None, 
                   include_top=False)

x_model.load_weights('xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[ ]:


x_model.summary()


# #### Finetuning
# 
# I started with a fully frozen model, then I included various additional layers. I found that freezing layers `1:85` resulted in quite efficient training, but I have trained the layers between 25 and 85 also for a few epochs.

# In[ ]:


print((x_model.layers[85]).name)
print((x_model.layers[25]).name)
print((x_model.layers[15]).name)


# In[ ]:


for layer in x_model.layers:
    layer.trainable = True

for layer in x_model.layers[:85]:
    layer.trainable = False   
    
x_model.summary()


# #### Generalized mean pool

# In[ ]:


gm_exp = tf.Variable(3., dtype=tf.float32)
def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                           axis=[1,2], 
                           keepdims=False)+1.e-8)**(1./gm_exp)
    return pool


# #### The top model

# In[ ]:


X_feat = Input(x_model.output_shape[1:])

lambda_layer = Lambda(generalized_mean_pool_2d)
lambda_layer.trainable_weights.extend([gm_exp])
X = lambda_layer(X_feat)
X = Dropout(0.05)(X)
X = Activation('relu')(X)
X = Dense(n_cat, activation='softmax')(X)

top_model = Model(inputs=X_feat, outputs=X)
top_model.summary()


# In[ ]:


X_image = Input(list(input_shape) + [3])

X_f = x_model(X_image)
X_f = top_model(X_f)

model = Model(inputs=X_image, outputs=X_f)
model.summary()


# #### Custom loss function
# 
# Individual losses are reweighted on each batch, but each output neuron will still always see a binary cross-entropy loss. In other words, the learning rate is simply higher for the most confident predictions.

# In[ ]:


def get_custom_loss(rank_weight=1., epsilon=1.e-9):
    def custom_loss(y_t, y_p):
        losses = tf.reduce_sum(-y_t*tf.log(y_p+epsilon) - (1.-y_t)*tf.log(1.-y_p+epsilon), 
                               axis=-1)
        
        pred_idx = tf.argmax(y_p, axis=-1)
        
        mask = tf.one_hot(pred_idx, 
                          depth=y_p.shape[1], 
                          dtype=tf.bool, 
                          on_value=True, 
                          off_value=False)
        pred_cat = tf.boolean_mask(y_p, mask)
        y_t_cat = tf.boolean_mask(y_t, mask)
        
        n_pred = tf.shape(pred_cat)[0]
        _, ranks = tf.nn.top_k(pred_cat, k=n_pred)
        
        ranks = tf.cast(n_pred-ranks, tf.float32)/tf.cast(n_pred, tf.float32)*rank_weight
        rank_losses = ranks*(-y_t_cat*tf.log(pred_cat+epsilon)
                             -(1.-y_t_cat)*tf.log(1.-pred_cat+epsilon))        
        
        return rank_losses + losses
    return custom_loss


# #### Additional metric
# 
# The GAP is estimated by calculating it on each batch during training.

# In[ ]:


def batch_GAP(y_t, y_p):
    pred_cat = tf.argmax(y_p, axis=-1)    
    y_t_cat = tf.argmax(y_t, axis=-1) * tf.cast(
        tf.reduce_sum(y_t, axis=-1), tf.int64)
    
    n_pred = tf.shape(pred_cat)[0]
    is_c = tf.cast(tf.equal(pred_cat, y_t_cat), tf.float32)

    GAP = tf.reduce_mean(
          tf.cumsum(is_c) * is_c / tf.cast(
              tf.range(1, n_pred + 1), 
              dtype=tf.float32))
    
    return GAP


# This is just a reweighting to yield larger numbers for the loss..

# In[ ]:


def binary_crossentropy_n_cat(y_t, y_p):
    return keras.metrics.binary_crossentropy(y_t, y_p) * n_cat


# #### Training
# 
# I manually decreased the learning rate during training, starting at about 0.001 for training the `top_model` (on a larger `batch_size` of 128 or so).

# In[ ]:


opt = Adam(lr=0.0001)
loss = get_custom_loss(1.0)
#loss='categorical_crossentropy'
#loss='binary_crossentropy'
model.compile(loss=loss, 
              optimizer=opt, 
              metrics=[binary_crossentropy_n_cat, 'accuracy', batch_GAP])


# In[ ]:


checkpoint1 = ModelCheckpoint('dd_checkpoint-1.h5', 
                              period=1, 
                              verbose=1, 
                              save_weights_only=True)
checkpoint2 = ModelCheckpoint('dd_checkpoint-2.h5', 
                              period=1, 
                              verbose=1, 
                              save_weights_only=True)
checkpoint3 = ModelCheckpoint('dd_checkpoint-3-best.h5', 
                              period=1, 
                              verbose=1, 
                              monitor='loss', 
                              save_best_only=True, 
                              save_weights_only=True)


# In[ ]:


#model.load_weights('dd_6_best_so_far.h5')


# In[ ]:


K.set_value(model.optimizer.lr, 0.0000003)


# #### Training

# In[ ]:


model.fit_generator(train_generator, 
                    steps_per_epoch=len(train_info) / batch_size / 8, 
                    epochs=50, 
                    callbacks=[checkpoint1, checkpoint2, checkpoint3])


# In[ ]:


model.save_weights('dd_1.h5')


# #### Some evaluations

# In[ ]:


K.eval(gm_exp)


# In[ ]:


print(model.history.history['loss'])


# In[ ]:


plt.plot(model.history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')


# In[ ]:


plt.plot(model.history.history['batch_GAP'])
plt.xlabel('epoch')
plt.ylabel('batch_GAP')


# In[ ]:


plt.plot(model.history.history['acc'])
plt.xlabel('epoch')
plt.ylabel('acc')


# #### Validation and prediciton

# In[ ]:


def predict(info_imgs, load_n_images=1024):
    n = len(info_imgs)
    max_p = np.zeros(n)
    pred = np.zeros(n)
    
    for ind in range(0,len(info_imgs),load_n_images):
        try:
            imgs = info_imgs[ind:ind+load_n_images]
        except:
            imgs = info_imgs[ind:]
        
        imgs = preprocess_input(imgs)
        proba = model.predict(imgs, batch_size=batch_size_predict)
        
        pred_i = np.argmax(proba, axis=1)
        max_p[ind:(ind + load_n_images)] = proba[np.arange(len(pred_i)),pred_i]
        pred[ind:(ind + load_n_images)] = label_encoder.inverse_transform(pred_i)
        
        print(ind, '/', len(info_imgs), '  -->', pred[ind], max_p[ind])

    print(len(info_imgs), '/', len(info_imgs), '  -->', pred[-1], max_p[-1])
    
    return pred, max_p


# In[ ]:


def validate(info_imgs, load_n_images=1024):
        
    pred, max_p = predict(info_imgs, load_n_images=load_n_images)
    
    y = y_val
    binary_acc = accuracy_score(y, pred)

    sort_ind = np.argsort(max_p)[::-1]

    pred = pred[sort_ind]
    y_true = y[sort_ind]

    GAP = np.sum(np.cumsum(pred == y_true) * (pred == y_true) / np.arange(1, len(y_true) + 1)) / np.sum(y_true >= 0.)

    print("accuracy:", binary_acc, "\n ")
    print("*** GAP:", GAP, "***")
    
    return binary_acc, GAP


# Validate only on landmark images

# In[ ]:


from keras.applications.vgg16 import preprocess_input, decode_predictions
preds = model.predict(X_val)
print ('Predicted:', preds[:1])


# In[ ]:


dev_binary_acc, dev_GAP = validate(X_val, 1024)


# Validate on landmark and non-landmark images

# In[ ]:





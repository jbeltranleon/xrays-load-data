
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import os
get_ipython().system('pip install keras==2.1.3')
import os


# In[ ]:


# reading the data
data = pd.read_pickle("../input/chest-xray-data-multi14/data-chest-x-ray-multilabel-14.plk")
print(data.shape)
data.head(2)


# In[ ]:


from sklearn.model_selection import train_test_split
# same proportion of values provided to parameter stratify
train_df, valid_df = train_test_split(data, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = data['Finding Labels'].map(lambda x: x[:4]))

print('train', train_df.shape[0], 'validation', valid_df.shape[0])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

# New Size
IMG_SIZE = (224, 224,)

# Image Parameters
core_idg = ImageDataGenerator(
                                samplewise_std_normalization=True,
                                horizontal_flip = True, 
                                vertical_flip = False,
                            )


# In[ ]:


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('Base dir ===>', base_dir)
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    print('df_gen: ', df_gen, '\n')
    
    df_gen.filenames = in_df[path_col].values
    print(in_df.shape)
    print('df_gen.filenames: ',len(df_gen.filenames), df_gen.filenames[0], '...\n')
    
    df_gen.classes = np.stack(in_df[y_col].values)
    print(type(df_gen.classes), 'df_gen.classes: ', df_gen.classes[0], '...\n')
    
    df_gen.samples = in_df.shape[0]
    print(type(df_gen.samples), 'df_gen.samples: ', df_gen.samples, '\n')
    
    df_gen.n = in_df.shape[0]
    print('df_gen.n: ', df_gen.n, '\n')
    
    df_gen._set_index_array()
    print('df_gen._set_index_array: ', df_gen._set_index_array(), '\n')
    df_gen.directory = '' # since we have the full path
    print('df_gen.directory: ', df_gen.directory, '\n')
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# In[ ]:


print('################# TRAIN ############################ \n\n')
# y_col add the disease_vec to the target column for a multilabel prediction

# Iterator to guide the flow of read
train_gen = flow_from_dataframe(img_data_gen=core_idg, 
                                in_df= train_df, 
                                path_col = 'path',
                                y_col = 'disease_vec', 
                                target_size = IMG_SIZE,
                                batch_size = 32)

print(len(train_gen))
print(train_gen)

print('\n\n################# VAL ############################ \n\n')
valid_gen = flow_from_dataframe(img_data_gen=core_idg, 
                                in_df=valid_df,
                                path_col = 'path',
                                y_col = 'disease_vec', 
                                target_size = IMG_SIZE,
                                batch_size = 256) # we can use much larger batches for evaluation

print(len(valid_gen))

# used a fixed dataset for evaluating the algorithm
print('\n\n################# TEST ############################ \n\n')
test_X, test_Y = next(flow_from_dataframe(img_data_gen=core_idg, 
                                          in_df=valid_df, 
                                          path_col = 'path',
                                          y_col = 'disease_vec',
                                          target_size = IMG_SIZE,
                                          batch_size = 1024)) # one big batch

print(len(test_X), len(test_Y))


# In[ ]:


#image array
t_x, t_y = next(train_gen)

#This is the image size!
print(f'Array (image) size: {len(t_x[0])} x {len(t_x[0][0])}')

type(t_x), type(t_y)


# In[ ]:


from itertools import chain

all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
MIN_CASES = 0
all_labels = [c_label for c_label in all_labels if data[c_label].sum()>MIN_CASES]


# In[ ]:


# 16 Spaces for paint images
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    #Paint each image
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    #Set title for each image
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                             if n_score>0.5]))
    #without axis
    c_ax.axis('off')


# ### Create a simple model
# Here we make a simple model to train using DenseNet121 as a base and then adding a GAP layer (Flatten could also be added), dropout, and a fully-connected layer to calculate specific features

# In[ ]:


# Application Arch
from keras.applications.densenet import DenseNet121
# Extra Layers
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
# Create as hamburger
from keras.models import Sequential


# In[ ]:


# input_shape is the dimensions in the first layer
#densenet121_model_top = DenseNet121(input_shape =  t_x.shape[1:], 
#                                 include_top = True, weights = None)
#densenet121_model_top.summary()


# In[ ]:


#densenet121_model_top.get_config()


# In[ ]:


# input_shape is the dimensions in the first layer
densenet121_model = DenseNet121(input_shape =  t_x.shape[1:], 
                                 include_top = False, weights = None)
#densenet121_model.summary()


# In[ ]:


#Entry point
multi_disease_model = Sequential()
#add all DenseNet121 with top
multi_disease_model.add(densenet121_model)
multi_disease_model.summary()


# In[ ]:


multi_disease_model.add(GlobalAveragePooling2D())
#multi_disease_model.add(Dropout(0.5))
#multi_disease_model.add(Dense(512))
#multi_disease_model.add(Dropout(0.5))

#Activation layer
#multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_model.add(Dense(units=len(all_labels), activation='softmax', kernel_initializer='VarianceScaling'))

#### IMPORTANT
#### configure the learning process

from keras.optimizers import Adam
learning_rate = 0.0001

multi_disease_model.compile(optimizer=Adam(lr=learning_rate), loss = 'binary_crossentropy',
                           metrics = ['accuracy','binary_accuracy', 'mae'])
multi_disease_model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import keras.callbacks as kcall
weight_path="{}_weights.best.hdf5".format('xray_class')
weight_path


# In[ ]:


class LossHistory(kcall.Callback):
    def on_train_begin(self, logs={}):
        #Batch
        self.batch_losses = []
        self.batch_acc = []
        #Epochs
        self.epochs_losses = []
        self.epochs_acc = []
        self.epochs_val_losses = []
        self.epochs_val_acc = []
        
    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))
        self.batch_acc.append(logs.get('acc'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.epochs_losses.append(logs.get('loss'))
        self.epochs_acc.append(logs.get('acc'))
        self.epochs_val_losses.append(logs.get('val_loss'))
        self.epochs_val_acc.append(logs.get('val_acc'))


# In[ ]:


history = LossHistory()

# To save the bests weigths
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10)

callbacks_list = [checkpoint, early, history]
callbacks_list


# In[ ]:


multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 50, 
                                  callbacks = callbacks_list)


# In[ ]:


# Space for the first row that contain 2 graphs
plt.figure(figsize=[16,8])
### 1 ###

######### Plots of losses

# Space
plt.subplot(2, 2, 1)

# batch_losses line
plt.plot(history.batch_losses,'b--',label='Train',alpha=0.7)

# Axis labels
plt.xlabel('# of batches trained')
plt.ylabel('Training loss')

# Title
plt.title('1) Training loss vs batches trained')

#Small square with convension on the rigth side
plt.legend()

plt.ylim(0,1)
plt.grid(True)

### 2 ###

# Space
plt.subplot(2, 2, 2)

# epochs_losses line
plt.plot(history.epochs_losses,'b--',label='Train',alpha=0.7)

# epochs_val_losses line
plt.plot(history.epochs_val_losses,'r-.',label='Val', alpha=0.7)

# Axis labels
plt.xlabel('# of epochs trained')
plt.ylabel('Training loss')

# Title
plt.title('2) Training loss vs epochs trained')

#Small square with convension on the rigth side
plt.legend()

plt.ylim(0,0.5)
plt.grid(True)

### 3 ###

######### Plots of acc

# Space for the first row that contain 2 graphs
plt.figure(figsize=[16,8])

# Space
plt.subplot(2, 2, 1)

# batch_acc line
plt.plot(history.batch_acc,'b--',label= 'Train', alpha=0.7)

# Axis labels
plt.xlabel('# of batches trained')
plt.ylabel('Training accuracy')

# Title
plt.title('3) Training accuracy vs batches trained')

#Small square with convension on the left side
plt.legend(loc=3)

plt.ylim(0,1.1)
plt.grid(True)

### 4 ###

# Space
plt.subplot(2, 2, 2)

# epochs_acc line
plt.plot(history.epochs_acc,'b--',label= 'Train', alpha=0.7)

# epochs_val_acc line
plt.plot(history.epochs_val_acc,'r-.',label= 'Val', alpha=0.7)

# Axis labels
plt.xlabel('# of epochs trained')
plt.ylabel('Training accuracy')

# Title
plt.title('4) Training accuracy vs epochs trained')

#Small square with convension on the left side
plt.legend(loc=3)

plt.ylim(0.5,1)
plt.grid(True)


# ### Check Output
# Here we see how many positive examples we have of each category

# In[ ]:


for c_label, s_count in zip(all_labels, 100*np.mean(test_Y,0)):
    print('%s: %2.2f%%' % (c_label, s_count))


# In[ ]:


pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)


# ### ROC Curves
# While a very oversimplified metric, we can show the ROC curve for each metric

# In[ ]:


from sklearn.metrics import roc_curve, auc
# Space
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    #Points to graph
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    
#convention
c_ax.legend()

#Labels
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')

# Save as a png
fig.savefig('barely_trained_net.png')


# ### Continued Training
# Now we do a much longer training process to see how the results improve

# In[ ]:


# Prev values: steps_per_epoch=100, epochs = 10
multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch = 100,
                                  validation_data =  (test_X, test_Y), 
                                  epochs = 30, 
                                  callbacks = callbacks_list)


# In[ ]:


# load the best weights
multi_disease_model.load_weights(weight_path)


# In[ ]:


pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)


# In[ ]:


# Space for the first row that contain 2 graphs
plt.figure(figsize=[16,8])
### 1 ###

######### Plots of losses

# Space
plt.subplot(2, 2, 1)

# batch_losses line
plt.plot(history.batch_losses,'b--',label='Train',alpha=0.7)

# Axis labels
plt.xlabel('# of batches trained')
plt.ylabel('Training loss')

# Title
plt.title('1) Training loss vs batches trained')

#Small square with convension on the rigth side
plt.legend()

plt.ylim(0,1)
plt.grid(True)

### 2 ###

# Space
plt.subplot(2, 2, 2)

# epochs_losses line
plt.plot(history.epochs_losses,'b--',label='Train',alpha=0.7)

# epochs_val_losses line
plt.plot(history.epochs_val_losses,'r-.',label='Val', alpha=0.7)

# Axis labels
plt.xlabel('# of epochs trained')
plt.ylabel('Training loss')

# Title
plt.title('2) Training loss vs epochs trained')

#Small square with convension on the rigth side
plt.legend()

plt.ylim(0,0.5)
plt.grid(True)

### 3 ###

######### Plots of acc

# Space for the first row that contain 2 graphs
plt.figure(figsize=[16,8])

# Space
plt.subplot(2, 2, 1)

# batch_acc line
plt.plot(history.batch_acc,'b--',label= 'Train', alpha=0.7)

# Axis labels
plt.xlabel('# of batches trained')
plt.ylabel('Training accuracy')

# Title
plt.title('3) Training accuracy vs batches trained')

#Small square with convension on the left side
plt.legend(loc=3)

plt.ylim(0,1.1)
plt.grid(True)

### 4 ###

# Space
plt.subplot(2, 2, 2)

# epochs_acc line
plt.plot(history.epochs_acc,'b--',label= 'Train', alpha=0.7)

# epochs_val_acc line
plt.plot(history.epochs_val_acc,'r-.',label= 'Val', alpha=0.7)

# Axis labels
plt.xlabel('# of epochs trained')
plt.ylabel('Training accuracy')

# Title
plt.title('4) Training accuracy vs epochs trained')

#Small square with convension on the left side
plt.legend(loc=3)

plt.ylim(0.5,1)
plt.grid(True)


# In[ ]:


# look at how often the algorithm predicts certain diagnoses 
for c_label, p_count, t_count in zip(all_labels, 
                                     100*np.mean(pred_Y,0), 
                                     100*np.mean(test_Y,0)):
    print(f'{c_label}: Dx: {t_count:.2f}%, PDx: {p_count:.2f}%')


# In[ ]:


# Space
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    #Points to graph
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    
#convention
c_ax.legend()

#Labels
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')

# Save as a png
fig.savefig('barely_trained_net.png')


# ### Show a few images and associated predictions
# 

# In[ ]:


row = 20
print(*[f'{i}' for i in range(14)], sep='\t')
# test_Y
print(*test_Y[row], sep='\t', end='\tTrue\n')

sumatoria = np.sum(pred_Y[row])
print(f'sumatoria: {sumatoria}')

# pred_Y
pred = [round(item, 2) for item in pred_Y[row]]
print(*pred, sep='\t', end='\tPredict\n\n')

print(*[f'{i} \t ==>\t {l}' for i,l in enumerate(all_labels)], sep='\n')


# In[ ]:


# Pred_y of a single row image labels
pred_Y[660]


# In[ ]:


sickest_idx = np.argsort(np.sum(test_Y, 1)<1)

#Space of images
fig, m_axs = plt.subplots(10, 4, figsize = (16, 32))

# Padding
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
counter = 0

for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    
    # Image show
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    
    stat_str = [n_class[:4] for n_class, n_score in zip(all_labels, test_Y[idx]) if n_score>0.5]
    
    # Labels of the firts image
    counter += 1
    if counter == 1:
        print(f'Labels of the firts image: {stat_str}')
        
    # Building the labels
    pred_str = [f'{n_class[:4]}:{p_score*100:.2f}%'
                for n_class, n_score, p_score 
                in zip(all_labels,test_Y[idx],pred_Y[idx]) 
                if (n_score>0.5) or (p_score>0.5)]
    
    c_ax.set_title(f'Index {idx}, Labels: '+', '.join(stat_str)+'\n Pred: '+', '.join(pred_str))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png')
    

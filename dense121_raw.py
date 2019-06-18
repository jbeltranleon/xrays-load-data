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
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from itertools import chain
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential


data = pd.read_pickle("../input/chest-xray-data-multi14/data-chest-x-ray-multilabel-14.plk")

train_df, valid_df = train_test_split(data, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = data['Finding Labels'].map(lambda x: x[:4]))

IMG_SIZE = (224, 224,)
core_idg = ImageDataGenerator(
                                samplewise_std_normalization=True,
                                horizontal_flip = True, 
                                vertical_flip = False,
                            )


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]    
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    
    return df_gen


train_gen = flow_from_dataframe(img_data_gen=core_idg, 
                                in_df= train_df, 
                                path_col = 'path',
                                y_col = 'disease_vec', 
                                target_size = IMG_SIZE,
                                batch_size = 32)

valid_gen = flow_from_dataframe(img_data_gen=core_idg, 
                                in_df=valid_df,
                                path_col = 'path',
                                y_col = 'disease_vec', 
                                target_size = IMG_SIZE,
                                batch_size = 256) # we can use much larger batches for evaluation

test_X, test_Y = next(flow_from_dataframe(img_data_gen=core_idg, 
                                          in_df=valid_df, 
                                          path_col = 'path',
                                          y_col = 'disease_vec',
                                          target_size = IMG_SIZE,
                                          batch_size = 1024)) # one big batch

t_x, t_y = next(train_gen)
all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
MIN_CASES = 0
all_labels = [c_label for c_label in all_labels if data[c_label].sum()>MIN_CASES]


fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    #Paint each image
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    #Set title for each image
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                             if n_score>0.5]))
    #without axis
    c_ax.axis('off')


densenet121_model = DenseNet121(input_shape =  t_x.shape[1:], 
                                 include_top = False, weights = None)







multi_disease_model = Sequential()

multi_disease_model.add(densenet121_model)
multi_disease_model.summary()





multi_disease_model.add(GlobalAveragePooling2D())






multi_disease_model.add(Dense(units=len(all_labels), activation='softmax', kernel_initializer='VarianceScaling'))




from keras.optimizers import Adam
learning_rate = 0.0001

multi_disease_model.compile(optimizer=Adam(lr=learning_rate), loss = 'binary_crossentropy',
                           metrics = ['accuracy','binary_accuracy', 'mae'])
multi_disease_model.summary()





from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import keras.callbacks as kcall
weight_path="{}_weights.best.hdf5".format('xray_class')
weight_path





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





history = LossHistory()


checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10)

callbacks_list = [checkpoint, early, history]
callbacks_list





multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 50, 
                                  callbacks = callbacks_list)






plt.figure(figsize=[16,8])





plt.subplot(2, 2, 1)


plt.plot(history.batch_losses,'b--',label='Train',alpha=0.7)


plt.xlabel('# of batches trained')
plt.ylabel('Training loss')


plt.title('1) Training loss vs batches trained')


plt.legend()

plt.ylim(0,1)
plt.grid(True)




plt.subplot(2, 2, 2)


plt.plot(history.epochs_losses,'b--',label='Train',alpha=0.7)


plt.plot(history.epochs_val_losses,'r-.',label='Val', alpha=0.7)


plt.xlabel('# of epochs trained')
plt.ylabel('Training loss')


plt.title('2) Training loss vs epochs trained')


plt.legend()

plt.ylim(0,0.5)
plt.grid(True)






plt.figure(figsize=[16,8])


plt.subplot(2, 2, 1)


plt.plot(history.batch_acc,'b--',label= 'Train', alpha=0.7)


plt.xlabel('# of batches trained')
plt.ylabel('Training accuracy')


plt.title('3) Training accuracy vs batches trained')


plt.legend(loc=3)

plt.ylim(0,1.1)
plt.grid(True)




plt.subplot(2, 2, 2)


plt.plot(history.epochs_acc,'b--',label= 'Train', alpha=0.7)


plt.plot(history.epochs_val_acc,'r-.',label= 'Val', alpha=0.7)


plt.xlabel('# of epochs trained')
plt.ylabel('Training accuracy')


plt.title('4) Training accuracy vs epochs trained')


plt.legend(loc=3)

plt.ylim(0.5,1)
plt.grid(True)








for c_label, s_count in zip(all_labels, 100*np.mean(test_Y,0)):
    





pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)








from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    #Points to graph
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    

c_ax.legend()


c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')


fig.savefig('barely_trained_net.png')









multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch = 100,
                                  validation_data =  (test_X, test_Y), 
                                  epochs = 30, 
                                  callbacks = callbacks_list)






multi_disease_model.load_weights(weight_path)





pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)






plt.figure(figsize=[16,8])





plt.subplot(2, 2, 1)


plt.plot(history.batch_losses,'b--',label='Train',alpha=0.7)


plt.xlabel('# of batches trained')
plt.ylabel('Training loss')


plt.title('1) Training loss vs batches trained')


plt.legend()

plt.ylim(0,1)
plt.grid(True)




plt.subplot(2, 2, 2)


plt.plot(history.epochs_losses,'b--',label='Train',alpha=0.7)


plt.plot(history.epochs_val_losses,'r-.',label='Val', alpha=0.7)


plt.xlabel('# of epochs trained')
plt.ylabel('Training loss')


plt.title('2) Training loss vs epochs trained')


plt.legend()

plt.ylim(0,0.5)
plt.grid(True)






plt.figure(figsize=[16,8])


plt.subplot(2, 2, 1)


plt.plot(history.batch_acc,'b--',label= 'Train', alpha=0.7)


plt.xlabel('# of batches trained')
plt.ylabel('Training accuracy')


plt.title('3) Training accuracy vs batches trained')


plt.legend(loc=3)

plt.ylim(0,1.1)
plt.grid(True)




plt.subplot(2, 2, 2)


plt.plot(history.epochs_acc,'b--',label= 'Train', alpha=0.7)


plt.plot(history.epochs_val_acc,'r-.',label= 'Val', alpha=0.7)


plt.xlabel('# of epochs trained')
plt.ylabel('Training accuracy')


plt.title('4) Training accuracy vs epochs trained')


plt.legend(loc=3)

plt.ylim(0.5,1)
plt.grid(True)






for c_label, p_count, t_count in zip(all_labels, 
                                     100*np.mean(pred_Y,0), 
                                     100*np.mean(test_Y,0)):
    






fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    #Points to graph
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    

c_ax.legend()


c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')


fig.savefig('barely_trained_net.png')








row = 20




sumatoria = np.sum(pred_Y[row])



pred = [round(item, 2) for item in pred_Y[row]]









pred_Y[660]





sickest_idx = np.argsort(np.sum(test_Y, 1)<1)


fig, m_axs = plt.subplots(10, 4, figsize = (16, 32))


fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
counter = 0

for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    
    # Image show
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    
    stat_str = [n_class[:4] for n_class, n_score in zip(all_labels, test_Y[idx]) if n_score>0.5]
    
    # Labels of the firts image
    counter += 1
    if counter == 1:
        
        
    # Building the labels
    pred_str = [f'{n_class[:4]}:{p_score*100:.2f}%'
                for n_class, n_score, p_score 
                in zip(all_labels,test_Y[idx],pred_Y[idx]) 
                if (n_score>0.5) or (p_score>0.5)]
    
    c_ax.set_title(f'Index {idx}, Labels: '+', '.join(stat_str)+'\n Pred: '+', '.join(pred_str))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png')
    

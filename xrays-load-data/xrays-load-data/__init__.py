import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os

def load_data_from_pkl(path):
    #"../input/chest-xray-data-multi14/data-chest-x-ray-multilabel-14.plk"
    data = pd.read_pickle(path)
    print(data.shape)
    return data

"""
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
all_labels = [c_label for c_label in all_labels if data[c_label].sum()>MIN_CASES]"""
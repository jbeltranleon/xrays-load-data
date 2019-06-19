import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from itertools import chain

def load_data_from_pkl(path):
    #"../input/chest-xray-data-multi14/data-chest-x-ray-multilabel-14.plk"
    data = pd.read_pickle(path)
    print(f"shape {data.shape}")
    return data


def do_train_test_split(data):
    # same proportion of values provided to parameter stratify
    train_df, valid_df = train_test_split(data, 
                                    test_size = 0.25, 
                                    random_state = 2018,
                                    stratify = data['Finding Labels'].map(lambda x: x[:4]))

    print('train', train_df.shape[0], 'validation', valid_df.shape[0])
    return train_df, valid_df


# New Size
IMG_SIZE = (224, 224,)

# Image Parameters
_core_idg = ImageDataGenerator(
                                samplewise_std_normalization=True,
                                horizontal_flip = True, 
                                vertical_flip = False,
                            )

def _flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
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

# y_col add the disease_vec to the target column for a multilabel prediction
# Iterator to guide the flow of read

def create_flow_of_images(train_df, valid_df):
    train_gen = _flow_from_dataframe(img_data_gen=_core_idg, 
                                    in_df= train_df, 
                                    path_col = 'path',
                                    y_col = 'disease_vec', 
                                    target_size = IMG_SIZE,
                                    batch_size = 32)

    valid_gen = _flow_from_dataframe(img_data_gen=_core_idg, 
                                    in_df=valid_df,
                                    path_col = 'path',
                                    y_col = 'disease_vec', 
                                    target_size = IMG_SIZE,
                                    batch_size = 256) 
                                    # we can use much larger batches for evaluation
    
    return train_gen, valid_gen


# used a fixed dataset for evaluating the algorithm
def create_flow_of_images_test(valid_df):
    test_X, test_Y = next(_flow_from_dataframe(img_data_gen=_core_idg, 
                                            in_df=valid_df, 
                                            path_col = 'path',
                                            y_col = 'disease_vec',
                                            target_size = IMG_SIZE,
                                            batch_size = 1024)) # one big batch

    print(len(test_X), len(test_Y))
    return test_X, test_Y

def get_all_labels(data):
    all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    MIN_CASES = 0
    all_labels = [c_label for c_label in all_labels if data[c_label].sum()>MIN_CASES]
    print(f"Labels{len(all_labels)}")
    return all_labels
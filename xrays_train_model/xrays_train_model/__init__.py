# ### Create a simple model
# Application Arch
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

# Extra Layers
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
# Create as hamburger
from keras.models import Sequential

from keras.optimizers import Adam

# Early Stopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import keras.callbacks as kcall
weight_path="{}_weights.best.hdf5".format('xray_class')
weight_path

# Adam(lr=learning_rate)
# binary_crossentropy
def create_model_dense_121(weights, t_x, all_labels, loss, optimizer_with_lr):
    # input_shape is the dimensions in the first layer
    densenet121_model = DenseNet121(input_shape =  t_x.shape[1:], 
                                    include_top = False, weights = weights)

    multi_disease_model = Sequential()
    multi_disease_model.add(densenet121_model)

    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dense(units=len(all_labels), 
                                activation='softmax', 
                                kernel_initializer='VarianceScaling'))

    multi_disease_model.compile(optimizer=optimizer_with_lr, loss=loss ,
                            metrics = ['accuracy','binary_accuracy', 'mae'])
    return multi_disease_model

def create_model_dense_169(weights, t_x, all_labels, loss, optimizer_with_lr):
    # input_shape is the dimensions in the first layer
    densenet169_model = DenseNet169(input_shape =  t_x.shape[1:], 
                                    include_top = False, weights = None)

    multi_disease_model = Sequential()
    multi_disease_model.add(densenet169_model)

    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dense(units=len(all_labels), 
                                activation='softmax', 
                                kernel_initializer='VarianceScaling'))

    multi_disease_model.compile(optimizer=optimizer_with_lr, loss=loss ,
                            metrics = ['accuracy','binary_accuracy', 'mae'])
    return multi_disease_model

def create_model_dense_201(weights, t_x, all_labels, loss, optimizer_with_lr):
    # input_shape is the dimensions in the first layer
    densenet169_model = DenseNet201(input_shape =  t_x.shape[1:], 
                                    include_top = False, weights = None)

    multi_disease_model = Sequential()
    multi_disease_model.add(densenet169_model)

    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dense(units=len(all_labels), 
                                activation='softmax', 
                                kernel_initializer='VarianceScaling'))

    multi_disease_model.compile(optimizer=optimizer_with_lr, loss=loss ,
                            metrics = ['accuracy','binary_accuracy', 'mae'])
    return multi_disease_model

def create_model_mobilenet(weights, t_x, all_labels, loss, optimizer_with_lr):
    # input_shape is the dimensions in the first layer
    mobilenet_model = MobileNet(input_shape =  t_x.shape[1:], 
                                    include_top = False, weights = weights)

    multi_disease_model = Sequential()
    multi_disease_model.add(mobilenet_model)

    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Reshape(target_shape=(1,1,1024)))
    multi_disease_model.add(Dropout(rate=0.001))
    multi_disease_model.add(Conv2D(filters=len(all_labels), kernel_size=(1, 1), padding='same', data_format='channels_last', activation='linear', kernel_initializer='VarianceScaling'))
    multi_disease_model.add(Activation('softmax'))
    multi_disease_model.add(Reshape(target_shape=(len(all_labels),)))

    multi_disease_model.compile(optimizer=optimizer_with_lr, loss=loss ,
                            metrics = ['accuracy','binary_accuracy', 'mae'])
    return multi_disease_model

def create_model_vgg16(weights, t_x, all_labels, loss, optimizer_with_lr):
    # input_shape is the dimensions in the first layer
    vgg16_model = VGG16(input_shape =  t_x.shape[1:], 
                                 include_top = False, weights = 'imagenet')

    multi_disease_model = Sequential()
    multi_disease_model.add(vgg16_model)

    multi_disease_model.add(Flatten())
    multi_disease_model.add(Dense(units=4096, activation='relu', kernel_initializer='VarianceScaling'))
    multi_disease_model.add(Dense(units=4096, activation='relu', kernel_initializer='VarianceScaling'))
    multi_disease_model.add(Dense(units=len(all_labels), activation='softmax', kernel_initializer='VarianceScaling'))

    multi_disease_model.compile(optimizer=optimizer_with_lr, loss=loss ,
                            metrics = ['accuracy','binary_accuracy', 'mae'])
    return multi_disease_model

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

def early_stopping():
    history = LossHistory()

    # To save the bests weigths
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, 
                                save_best_only=True, mode='min', save_weights_only = True)

    early = EarlyStopping(monitor="val_loss", 
                        mode="min", 
                        patience=10)

    callbacks_list = [checkpoint, early, history]

    return checkpoint, early, callbacks_list


# First train 50
# Second train 30
def train(multi_disease_model, epochs, train_gen, test_X, test_Y, callbacks_list):
    multi_disease_model.fit_generator(train_gen, 
                                    steps_per_epoch=100,
                                    validation_data = (test_X, test_Y), 
                                    epochs = epochs, 
                                    callbacks = callbacks_list, verbose=0)





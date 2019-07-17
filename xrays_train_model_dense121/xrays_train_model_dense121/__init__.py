# ### Create a simple model
# Application Arch
from keras.applications.densenet import DenseNet121
# Extra Layers
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
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
def create_model_dense_121(t_x, all_labels, loss, optimizer_with_lr):
    # input_shape is the dimensions in the first layer
    densenet121_model = DenseNet121(input_shape =  t_x.shape[1:], 
                                    include_top = False, weights = None)

    multi_disease_model = Sequential()
    #add all DenseNet121 with top
    multi_disease_model.add(densenet121_model)

    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dense(units=len(all_labels), 
                                activation='softmax', 
                                kernel_initializer='VarianceScaling'))

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





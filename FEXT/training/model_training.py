import os
import tensorflow as tf
from keras.utils import plot_model

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.generators import build_tensor_dataset
from FEXT.commons.utils.dataloader.serializer import DataSerializer
from FEXT.commons.utils.preprocessing import get_images_path, DataSplit
from FEXT.commons.utils.models.training import ModelTraining
from FEXT.commons.utils.models.autoencoder import FeXTAutoEncoder
from FEXT.commons.utils.models.callbacks import RealTimeHistory
from FEXT.commons.pathfinder import IMG_DATA_PATH, CHECKPOINT_PATH
import FEXT.commons.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------    
    # select a fraction of data for training
    images_paths = get_images_path(IMG_DATA_PATH, num_images=cnf.NUM_OF_SAMPLES) 
    splitter = DataSplit(images_dictionary=images_paths)
    serializer = DataSerializer()     

    # split data    
    train_data, validation_data, test_data = splitter.split_data()   

    # create subfolder for preprocessing data    
    model_folder_path = serializer.create_checkpoint_folder()
    serializer.save_preprocessed_data(train_data, validation_data, test_data, output_type='JSON')      

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators    
    trainer = ModelTraining()
    trainer.set_device()

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators    
    train_dataset = build_tensor_dataset(train_data, cnf.BATCH_SIZE, cnf.IMG_SHAPE, 
                                         augmentation=cnf.IMG_AUGMENT, shuffle=True)
    validation_dataset = build_tensor_dataset(validation_data, cnf.BATCH_SIZE, cnf.IMG_SHAPE, 
                                              augmentation=cnf.IMG_AUGMENT, shuffle=True)
    test_dataset = build_tensor_dataset(test_data, cnf.BATCH_SIZE, cnf.IMG_SHAPE, 
                                        augmentation=cnf.IMG_AUGMENT, shuffle=True)   

    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/ 
            
    print('FeXT training report\n')
    print('--------------------------------------------------------------------')    
    print(f'Number of train samples: {train_data.shape[0]}')
    print(f'Number of test samples:  {test_data.shape[0]}')  
    print(f'Picture shape:           {cnf.IMG_SHAPE}')   
    print(f'Batch size:              {cnf.BATCH_SIZE}')
    print(f'Epochs:                  {cnf.EPOCHS}\n')   

    # build the autoencoder model     
    autoencoder = FeXTAutoEncoder(cnf.LEARNING_RATE, cnf.IMG_SHAPE, 
                                  cnf.SEED, XLA_state=cnf.XLA_STATE)
    model = autoencoder.get_model(summary=True) 

    # generate graphviz plot fo the model layout    
    if cnf.SAVE_MODEL_PLOT:
        plot_path = os.path.join(model_folder_path, 'model_layout.png')       
        plot_model(model, to_file = plot_path, show_shapes = True, 
                show_layer_names = True, show_layer_activations = True, 
                expand_nested = True, rankdir = 'TB', dpi = 400)

    # initialize the real time history callback    -
    RTH_callback = RealTimeHistory(model_folder_path, validation=True)
    callbacks_list = [RTH_callback]

    # initialize tensorboard if requested    
    if cnf.USE_TENSORBOARD:
        log_path = os.path.join(model_folder_path, 'tensorboard')
        callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, 
                                                             histogram_freq=1))

    # training loop and save model at end of training    
    multiprocessing = cnf.NUM_PROCESSORS > 1
    training = model.fit(train_dataset, epochs=cnf.EPOCHS, validation_data=test_dataset, 
                        callbacks=callbacks_list, workers=cnf.NUM_PROCESSORS, 
                        use_multiprocessing=multiprocessing)

    model_files_path = os.path.join(model_folder_path, 'model')
    model.save(model_files_path, save_format='tf')
    print(f'\nTraining session is over. Model has been saved in folder {model_folder_name}')

    # save model parameters in json files    
    parameters = {'train_samples': cnf.TRAIN_SAMPLES,
                'test_samples': cnf.TEST_SAMPLES,
                'picture_shape' : cnf.IMG_SHAPE,                           
                'augmentation' : cnf.IMG_AUGMENT,              
                'batch_size' : cnf.BATCH_SIZE,
                'learning_rate' : cnf.LEARNING_RATE,
                'epochs' : cnf.EPOCHS,
                'seed' : cnf.SEED,
                'tensorboard' : cnf.USE_TENSORBOARD}

    save_model_parameters(parameters, model_folder_path)




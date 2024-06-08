import os
import sys
import tensorflow as tf
from keras.utils import plot_model

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.generators import DataGenerator, create_tf_dataset
from FEXT.commons.utils.preprocessing import model_savefolder, dataset_from_images
from FEXT.commons.utils.models import ModelTraining, FeXTAutoEncoder, save_model_parameters
from FEXT.commons.utils.callbacks import RealTimeHistory
from FEXT.commons.pathfinder import IMG_DATA_PATH, CHECKPOINT_PATH
import FEXT.commons.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD AND PREPROCESS DATA]
    #--------------------------------------------------------------------------    
    # find and assign images path
    total_samples = cnf.TRAIN_SAMPLES + cnf.TEST_SAMPLES
    df_images = dataset_from_images(IMG_DATA_PATH)

    # select a fraction of data for training
    df_images = df_images.sample(total_samples, random_state=36).reset_index(drop=True)

    # create train and test datasets
    test_data = df_images.sample(n=cnf.TEST_SAMPLES, random_state=cnf.SPLIT_SEED)
    train_data = df_images.drop(test_data.index)

    # create subfolder for preprocessing data    
    model_folder_path, model_folder_name  = model_savefolder(CHECKPOINT_PATH, 'FeXT')
    pp_path = os.path.join(model_folder_path, 'preprocessing')
    os.mkdir(pp_path) if not os.path.exists(pp_path) else None

    # save preprocessed data
    file_loc = os.path.join(pp_path, 'train_data.csv')  
    train_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
    file_loc = os.path.join(pp_path, 'test_data.csv')  
    test_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    train_data.drop(columns='name', inplace=True)
    test_data.drop(columns='name', inplace=True)

    # initialize training device 
    # allows changing device prior to initializing the generators    
    trainer = ModelTraining(seed=cnf.SEED)
    trainer.set_device(device=cnf.ML_DEVICE, use_mixed_precision=cnf.MIXED_PRECISION)

    # initialize the images generator for the train and test data, and create the 
    # tf.dataset according to batch shapes    
    train_generator = DataGenerator(train_data, cnf.BATCH_SIZE, cnf.IMG_SHAPE, 
                                    augmentation=cnf.IMG_AUGMENT, shuffle=True)
    test_generator = DataGenerator(test_data, cnf.BATCH_SIZE, cnf.IMG_SHAPE, 
                                augmentation=cnf.IMG_AUGMENT, shuffle=True)

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators 
    train_dataset = create_tf_dataset(train_generator)
    test_dataset = create_tf_dataset(test_generator)

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
    modelworker = FeXTAutoEncoder(cnf.LEARNING_RATE, cnf.IMG_SHAPE, 
                                  cnf.SEED, XLA_state=cnf.XLA_STATE)
    model = modelworker.get_model(summary=True) 

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




import os
import sys

# setting warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and components
#------------------------------------------------------------------------------
from utils.generators import dataloader
from utils.preprocessing import dataset_from_images
from utils.models import ModelTraining, FeXTAutoEncoder, model_savefolder, model_parameters
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
images_path = os.path.join(globpt.data_path, 'images')
cp_path = os.path.join(globpt.train_path, 'checkpoints')
os.mkdir(images_path) if not os.path.exists(images_path) else None
os.mkdir(cp_path) if not os.path.exists(cp_path) else None


# [RUN MAIN]
#------------------------------------------------------------------------------
if __name__ == '__main__':

    # [LOAD DATA AND ADD IMAGES PATHS TO DATASET]
    #--------------------------------------------------------------------------    
    # 1. find and assign images path, then select a fraction of data for training
    total_samples = cnf.num_train_samples + cnf.num_test_samples
    df_images = dataset_from_images(images_path)
    df_images = df_images.sample(total_samples, random_state=36).reset_index(drop=True)

    # 2. create train and test datasets
    test_data = df_images.sample(n=cnf.num_test_samples, random_state=cnf.split_seed)
    train_data = df_images.drop(test_data.index)

    # 3. create subfolder for preprocessing data    
    model_folder_path, model_folder_name = model_savefolder(cp_path, 'FeXT')
    pp_path = os.path.join(model_folder_path, 'preprocessing')
    os.mkdir(pp_path) if not os.path.exists(pp_path) else None

    # 4. save preprocessed data
    file_loc = os.path.join(pp_path, 'train_data.csv')  
    train_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
    file_loc = os.path.join(pp_path, 'test_data.csv')  
    test_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')

    # [DEFINE IMAGES GENERATOR AND BUILD DATALOADER]
    #--------------------------------------------------------------------------  
    train_data.drop(columns='name', inplace=True)
    test_data.drop(columns='name', inplace=True)

    # 1. initialize training device
    print('\nInitialize training device as per user configurations')
    trainer = ModelTraining(device=cnf.training_device, seed=cnf.seed, 
                            use_mixed_precision=cnf.use_mixed_precision)    

    # 2. initialize the images generator for the train and test data
    train_generator = dataloader(train_data, cnf.batch_size, cnf.picture_shape,
                                 shuffle=True, augmentation=cnf.augmentation,
                                 device=cnf.training_device)
    test_generator = dataloader(test_data, cnf.batch_size, cnf.picture_shape,
                                shuffle=True, augmentation=cnf.augmentation,
                                device=cnf.training_device)    

    # [TRAINING MODEL]
    #-------------------------------------------------------------------------- 
    # Setting callbacks and training routine for the features extraction model. 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/
    
    print(f'\nFeXT training report\n{"-" * 82}')    
    print(f'Number of train samples: {train_data.shape[0]}')
    print(f'Number of test samples:  {test_data.shape[0]}')
    print(f'Picture shape:           {cnf.picture_shape}')
    print(f'Kernel size:             {(3, 3)}')
    print(f'Batch size:              {cnf.batch_size}')
    print(f'Epochs:                  {cnf.epochs}')
  
    # 1. build the autoencoder model and print summary
    print('\nInitialise FeXT autoencoder\n')   
    model = FeXTAutoEncoder(cnf.seed)  
    model.print_summary()  

    # 2. training loop and save model at end of training    
    training = trainer.train_model(model, train_generator, test_generator, 
                                   cnf.epochs, cnf.learning_rate)
    
    model_subfolder = os.path.join(model_folder_path, 'model')
    os.mkdir(model_subfolder ) if not os.path.exists(model_subfolder ) else None
    trainer.save_model(model, model_subfolder)

    # 3. save model parameters in json files   
    print(f'Training session is over. Model has been saved in folder {model_folder_name}')
    parameters = {'train_samples': cnf.num_train_samples,
                  'test_samples': cnf.num_test_samples,
                  'picture_shape' : cnf.picture_shape,             
                  'kernel_size' : (3,3),              
                  'augmentation' : cnf.augmentation,              
                  'batch_size' : cnf.batch_size,
                  'learning_rate' : cnf.learning_rate,
                  'epochs' : cnf.epochs,
                  'seed' : cnf.seed,
                  'tensorboard' : cnf.use_tensorboard}

    model_parameters(parameters, model_folder_path)
                                


# generate graphviz plot fo the model layout
#------------------------------------------------------------------------------
# if cnf.generate_model_graph==True:
#     plot_path = os.path.join(model_folder_path, 'model_layout.png')       
#     plot_model(model, to_file = plot_path, show_shapes = True, 
#                show_layer_names = True, show_layer_activations = True, 
#                expand_nested = True, rankdir = 'TB', dpi = 400)

# initialize the real time history callback
#------------------------------------------------------------------------------
# RTH_callback = RealTimeHistory(model_folder_path, validation=True)
# callbacks_list = [RTH_callback]

# initialize tensorboard if requested
#------------------------------------------------------------------------------
# if cnf.use_tensorboard:
#     log_path = os.path.join(model_folder_path, 'tensorboard')
#     callbacks_list.append(tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1))






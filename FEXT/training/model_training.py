# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.generators import build_tensor_dataset
from FEXT.commons.utils.dataloader.serializer import get_images_path, DataSerializer, ModelSerializer
from FEXT.commons.utils.preprocessing import DataSplit
from FEXT.commons.utils.models.training import ModelTraining
from FEXT.commons.utils.models.autoencoder import FeXTAutoEncoder
from FEXT.commons.configurations import IMG_SHAPE, BATCH_SIZE, EPOCHS, SAMPLE_SIZE
from FEXT.commons.pathfinder import VAL_PATH, IMG_DATA_PATH


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------    
    # select a fraction of data for training    
    images_paths = get_images_path(IMG_DATA_PATH, sample_size=SAMPLE_SIZE)    

    # split data
    print('\nPreparing dataset of images based on splitting sizes')  
    splitter = DataSplit(images_paths)     
    train_data, validation_data, test_data = splitter.split_data()   

    # create subfolder for preprocessing data    
    print('Saving images path references') 
    dataserializer = DataSerializer()
    model_folder_path = dataserializer.create_checkpoint_folder()
    dataserializer.save_preprocessed_data(train_data, validation_data, 
                                          test_data, model_folder_path)      

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators
    print('Building autoencoder model and data loaders\n')     
    trainer = ModelTraining()
    trainer.set_device()

    # initialize the TensorDataSet class with the generator instances
    # create the tf.datasets using the previously initialized generators    
    train_dataset = build_tensor_dataset(train_data)
    validation_dataset = build_tensor_dataset(validation_data)
    test_dataset = build_tensor_dataset(test_data)   

    # 3. [TRAINING MODEL]  
    #--------------------------------------------------------------------------  
    # Setting callbacks and training routine for the features extraction model 
    # use command prompt on the model folder and (upon activating environment), 
    # use the bash command: python -m tensorboard.main --logdir tensorboard/ 
            
    print('\nFeXT training report')
    print('--------------------------------------------------------------------')    
    print(f'Number of train samples:       {len(train_data)}')
    print(f'Number of validation samples:  {len(validation_data)}')
    print(f'Number of test samples:        {len(test_data)}')  
    print(f'Picture shape:                 {IMG_SHAPE}')   
    print(f'Batch size:                    {BATCH_SIZE}')
    print(f'Epochs:                        {EPOCHS}\n')  
    print('--------------------------------------------------------------------')  

    # build the autoencoder model     
    autoencoder = FeXTAutoEncoder()
    model = autoencoder.get_model(summary=True) 

    # generate graphviz plot fo the model layout 
    modelserializer = ModelSerializer()     
    modelserializer.save_model_plot(model, model_folder_path)              

    # perform training and save model at the end
    trainer.training_session(model, train_dataset, validation_dataset, model_folder_path)




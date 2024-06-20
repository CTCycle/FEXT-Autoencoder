# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from FEXT.commons.utils.dataloader.generators import build_tensor_dataset
from FEXT.commons.utils.dataloader.serializer import DataSerializer, ModelSerializer
from FEXT.commons.utils.preprocessing import get_images_path, DataSplit
from FEXT.commons.utils.models.training import ModelTraining
from FEXT.commons.utils.models.autoencoder import FeXTAutoEncoder
from FEXT.commons.configurations import IMG_SHAPE, BATCH_SIZE, EPOCHS


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD AND SPLIT DATA]
    #--------------------------------------------------------------------------    
    # select a fraction of data for training
    images_paths = get_images_path() 
    splitter = DataSplit(images_paths)
    dataserializer = DataSerializer()  
    modelserializer = ModelSerializer()  

    # split data    
    train_data, validation_data, test_data = splitter.split_data()   

    # create subfolder for preprocessing data    
    model_folder_path = dataserializer.create_checkpoint_folder()
    dataserializer.save_preprocessed_data(train_data, validation_data, test_data, output_type='JSON')      

    # 2. [DEFINE IMAGES GENERATOR AND BUILD TF.DATASET]
    #--------------------------------------------------------------------------
    # initialize training device 
    # allows changing device prior to initializing the generators    
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
            
    print('FeXT training report\n')
    print('--------------------------------------------------------------------')    
    print(f'Number of train samples:       {len(train_data)}')
    print(f'Number of validation samples:  {len(validation_data)}')
    print(f'Number of test samples:        {len(test_data)}')  
    print(f'Picture shape:                 {IMG_SHAPE}')   
    print(f'Batch size:                    {BATCH_SIZE}')
    print(f'Epochs:                        {EPOCHS}\n')   

    # build the autoencoder model     
    autoencoder = FeXTAutoEncoder()
    model = autoencoder.get_model(summary=True) 

    # generate graphviz plot fo the model layout    
    modelserializer.save_model_plot(model, model_folder_path)              

    # perform training and save model at the end
    trainer.training_session(model, train_dataset, validation_dataset, model_folder_path)




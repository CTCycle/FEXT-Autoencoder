from FEXT.commons.logger import logger


###############################################################################
def log_training_report(train_data, validation_data, config : dict):
    logger.info('--------------------------------------------------------------')
    logger.info('FeXT training report')
    logger.info('--------------------------------------------------------------')    
    logger.info(f'Number of train samples:       {len(train_data)}')
    logger.info(f'Number of validation samples:  {len(validation_data)}')    
    for key, value in config.items():
        if isinstance(value, dict) and ('validation' not in key and 'inference' not in key):
            for sub_key, sub_value in value.items():                             
                if isinstance(sub_value, dict):
                    for inner_key, inner_value in sub_value.items():
                        logger.info(f'{key} - {sub_key} - {inner_key}: {inner_value}')
                else:
                    logger.info(f'{key} - {sub_key}: {sub_value}')
        elif 'validation' not in key and 'inference' not in key:
            logger.info(f'{key}: {value}')

    logger.info('--------------------------------------------------------------\n')

        

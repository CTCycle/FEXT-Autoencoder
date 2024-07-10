import os
import logging
import logging.config
from datetime import datetime

from FEXT.commons.constants import LOGS_PATH

# Generate timestamp for the log filename
current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(LOGS_PATH, f'BGR_{current_timestamp}.log')

# Define custom filter class for logging behavior
class InfoOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

# Define logger configuration
LOG_CONFIG =  {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%d-%m-%Y %H:%M:%S'
        },        
        'minimal': {
            'format': '%(message)s',
        },
    },
    'filters': {
        'info_only_filter': {
            '()': InfoOnlyFilter,
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'minimal',
            'filters': ['info_only_filter'],            
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': log_filename,
            'mode': 'a'  
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'file'],
    },
}

# override logger configuration and load the latter
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger()
logger.debug('Logger has been loaded')

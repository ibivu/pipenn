import logging.config
import json

LOG_CONF_FILE = '../config/logging.json'

class PPILoggerCls(object):
    @classmethod
    def initLogger(cls, loggerName):
        with open(LOG_CONF_FILE, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logger = logging.getLogger(loggerName)    
        
        return logger
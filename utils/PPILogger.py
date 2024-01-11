import logging.config
import json

class PPILoggerCls(object):
    @classmethod
    def initLogger(cls, loggerName, pipennHome):
        LOG_CONF_FILE = pipennHome + '/config/logging.json'
        with open(LOG_CONF_FILE, 'rt') as f:
            config = json.load(f)
            
        for handler in config['handlers'].values():
            #print("%%%%%%%%%%%%% handler: ", handler)
            if 'filename' in handler:
                fileName = handler['filename']
                handler['filename'] = pipennHome + "/logs/" + fileName
                    
        logging.config.dictConfig(config)
        logger = logging.getLogger(loggerName)    
        
        return logger
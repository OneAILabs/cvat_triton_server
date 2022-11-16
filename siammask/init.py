from .utils.config_helper import load_config
from .custom import Custom
    
config = load_config()
siammask = Custom(anchors=config['anchors'])


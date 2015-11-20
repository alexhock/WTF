__author__ = 'ah14aeb'
from configobj import ConfigObj, ConfigObjError

def get_config(argv):

    config = None
    config_file_name = "./config.ini"

    if len(argv) > 1:
        config_file_name = argv[1]

    try:
        config = ConfigObj(config_file_name, file_error=True)
        config.interpolation = True
    except (ConfigObjError, IOError), e:
        print("Error could not read {0}: {1}".format(config_file_name, e))
        exit(1)

    return config

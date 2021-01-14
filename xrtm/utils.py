class ConfigException(Exception):
    pass

def _requires(key:str, cfg:dict, msg:str):
    if key not in cfg:
        raise ConfigException( msg )

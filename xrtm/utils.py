import re
from typing import Iterator

class ConfigException(Exception):
    pass

def _requires(key:str, cfg:dict, msg:str):
    if key not in cfg:
        raise ConfigException( msg )

def gen_timesteps(step:float, t=0.) -> Iterator[int]:
    while True:
        t += step
        yield round(t)

UB = r'(?:\[(\d*)\])?(?:-(\d))?(?:\.\w*)$'

def user_buffer_from_path(fp):
    m = re.search(UB, str(fp))
    return m.groups() if m else None, None
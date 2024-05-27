import logging, sys, os

from .run import Run

_MASTER_LOG = "aiomic"
_LOGS = set()
_FORMAT = '[%(asctime)s | %(name)-15s | %(levelname)-7s | %(message)s]'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

### Default Aiomic logging.
def init(name=_MASTER_LOG, level="DEBUG", filename=None, stdout=True):
    global _MASTER_LOG, _LOGS, _FORMAT, _DATE_FORMAT

    is_main = (name==_MASTER_LOG)
    if not is_main and _MASTER_LOG not in _LOGS:
        # Master log not initialized - do so
        init(name=_MASTER_LOG)
    
    # Only init aiomic main and child logs through this function
    name = name if name.startswith(_MASTER_LOG) else _MASTER_LOG+"."+name

    if name in _LOGS:# and not update:
        log(name).warning(f"Tried to initialize {name}, but it was already initialized.") # Set 'update=True' to re-initialize.")
        return log(name)
    
    _LOGS.add(name)

    if filename is None and Run.is_remote():
        filename = f"./logs/aiomic/{name.replace('.','_')}.txt"
    
    fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    log_obj = logging.getLogger(name)
    log_obj.setLevel(level)

    if is_main and stdout and not Run.is_remote():
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        h.setFormatter(fmt)
        log_obj.addHandler(h)
    if filename is not None:
        # Create dir
        if len(filename.split("/"))>1:
            os.makedirs("/".join(filename.split("/")[:-1]), exist_ok=True)
        h = logging.FileHandler(filename)
        h.setLevel(level)
        h.setFormatter(fmt)
        log_obj.addHandler(h)

    log_obj.info(f"Initialized log, level = {level}")
    if filename is not None:
        log_obj.info(f"Writing to file: {filename}")
    
    return log_obj

def attach_callback(name, cb):
    raise Exception("Not implemented!")

def log(name=None):
    global _MASTER_LOG, _LOGS
    
    name = _MASTER_LOG if name is None else name
    name = name if name.startswith(_MASTER_LOG) else _MASTER_LOG+"."+name
    return logging.getLogger(name) if name in _LOGS else init(name=name)



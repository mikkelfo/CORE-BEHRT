import pandas as pd

from . import log

class InvalidTypeError(Exception):
    def __init__(self, tp, valid_tps, name=None):
        super().__init__(f"Invalid parameter type: expected on of {valid_tps}, but got type {tp}"+("" if name is None else f" for parameter '{name}'"))

class InvalidValueError(Exception):
    def __init__(self, valid_vals=None, invalid_vals=None, name=None):
        msg = "Invalid parameter value: "
        if invalid_vals is not None:
            msg += f"value(s) {invalid_vals} are not valid"
            if valid_vals is not None:
                msg += f", expected values: {valid_vals}"
        elif valid_vals is not None:
            msg += f"expected values: {valid_vals}"
        super().__init__(msg+("" if name is None else f" for parameter '{name}'"))

class InvalidLengthError(Exception):
    def __init__(self, ls, min_length, name=None):
        msg = "Invalid length: expected "
        msg += ("non-empty" if min_length==1 else f"length >= {min_length}")
        msg += f", but got {len(ls)}"
        super().__init__(msg+("" if name is None else f" for parameter '{name}'"))

class InvalidDataFrameError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

def validate(
        inp,
        valid_types=None,
        valid_values=None,
        min_length=0,
        index_name=None,
        required_cols=None,
        none_ok=False,
        valid_func=None,
        name=None,
        raise_ex=True,
        ):
    try:
        if none_ok and inp is None:
            return True

        # validate type
        tp = type(inp)
        if valid_types is not None and tp not in valid_types:
            raise InvalidTypeError(tp, valid_types, name=name)
        
        # validate value
        if tp in (list,set):
            if min_length is not None and len(inp)<min_length:
                raise InvalidLengthError(inp, min_length, name=name)
            if valid_values is not None:
                invalids = set(inp) - set(valid_values)
                if len(invalids)>0:
                    raise InvalidValueError(invalid_vals=invalids, name=name)
        elif tp == pd.DataFrame:
            if index_name is not None and inp.index.name != index_name:
                raise InvalidDataFrameError(f"Invalid index name: expected {index_name}, but got {inp.index.name}"+("" if name is None else f" for parameter {name}"))
            if required_cols is not None:
                missing_cols = set(required_cols)-set(inp.columns)
                if len(missing_cols) > 0:
                    raise InvalidDataFrameError(f"Missing columns: {missing_cols}"+("" if name is None else f" for parameter {name}"))
    
        # Custom validation
        if valid_func is not None and not valid_func(inp):
            raise InvalidValueError(invalid_vals=inp, name=name)

    except Exception as e:
        if raise_ex:
            raise e
        else:
            log.warning(f"Warning: {e.message}")
            return False
    return True








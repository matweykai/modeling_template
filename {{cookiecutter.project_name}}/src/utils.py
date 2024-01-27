import importlib
from typing import Any


def load_object(obj_path: str, default_obj_path: str = '') -> Any:
    """Loads any python object described in the string

    Args:
        obj_path (str): defines object that we should import
        default_obj_path (str, optional): default place from which we should import object. Defaults to ''.

    Raises:
        AttributeError: if obj doesn't exist or it's name is wrong

    Returns:
        Any: python object that you described in the obj_path
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    
    module_obj = importlib.import_module(obj_path)
    
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    
    return getattr(module_obj, obj_name)

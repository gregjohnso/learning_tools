import importlib
from typing import Any, Dict, List, Union


def load_object(name: str, kwargs: Union[Dict[str, Any], None]) -> Any:
    """
    Load an object from a string, and pass kwargs to the constructor.

    If kwargs contain an object definition, then recursively load the object.
    If kwargs is None, then the object is returned as is, e.g. if you want to load a function, or a classmethod.

    This function performs a breadth-first search, starting at leaf nodes and working up.
    It descends both lists and dictionaries. If it encounters a [name, kwargs] pair, it instantiates it.

    Args:
        name (str): The fully qualified name of the object to load.
        kwargs (Union[Dict[str, Any], None]): The keyword arguments to pass to the object's constructor.

    Returns:
        Any: The loaded and instantiated object.
    """
    module_name, object_name = name.rsplit(".", 1)

    # Import the module
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Could not import module {module_name}. Error: {e}")

    # Get the object from the module
    try:
        obj = getattr(module, object_name)
    except AttributeError as e:
        raise AttributeError(
            f"Could not find attribute {object_name} in module {module_name}. Error: {e}"
        )

    if kwargs is None:
        return obj
    else:
        queue: List[Union[Dict[str, Any], List[Any]]] = [kwargs]

        while queue:
            current = queue.pop(0)

            if isinstance(current, dict):
                for k, v in current.items():
                    if (
                        isinstance(v, dict)
                        and "name" in v
                        and "kwargs" in v
                        and len(v) == 2
                    ):
                        current[k] = load_object(**v)
                    elif isinstance(v, list):
                        queue.append(v)
            elif isinstance(current, list):
                for i, item in enumerate(current):
                    if (
                        isinstance(item, dict)
                        and "name" in item
                        and "kwargs" in item
                        and len(item) == 2
                    ):
                        current[i] = load_object(**item)
                    elif isinstance(item, list):
                        queue.append(item)

        try:
            return obj(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Could not instantiate {object_name} in module {module_name} with kwargs {kwargs}. Error: {e}"
            )

import typing

def is_typing_type(object, type):
    """
    Checks if an object is of a certain typing type, including nested types.
    
    This function supports simple types (like `int`, `str`), typing types 
    (like `List[int]`, `Tuple[str, int]`, `Dict[str, int]`), and nested typing 
    types (like `List[List[int]]`, `Tuple[List[str], int]`, `Dict[str, List[int]]`).

    Args:
        object: The object to check.
        type: The typing type to check against.

    Returns:
        bool: True if the object is of the specified type, False otherwise.

    Examples:
        >>> is_typing_type(1, int)
        True
        >>> is_typing_type([1, 2, 3], typing.List[int])
        True
        >>> is_typing_type([1, 2, 3], typing.List[str])
        False
        >>> is_typing_type([[1, 2], [3, 4]], typing.List[typing.List[int]])
        True
    """
    
    if hasattr(type, '__origin__'):
        origin = type.__origin__
        type_args = typing.get_args(type)

        if origin is list or origin is set:
            return all(is_typing_type(element, type_args[0]) for element in object)
        elif origin is dict:
            return all(is_typing_type(key, type_args[0]) and is_typing_type(value, type_args[1]) for key, value in object.items())
        elif origin is tuple:
            return all(is_typing_type(element, type_arg) for element, type_arg in zip(object, type_args))
    else:
        return isinstance(object, type)
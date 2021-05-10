import importlib


def load_class(full_class_string):
    """
    dynamically load a class from a string

    Parameters
    ----------
    full_class_string : str
        The full class string to the given python class.
        Example:
            my_project.my_module.my_class

    Returns
    -------
    python class
        PYthon class defined by the 'full_class_string'
    """

    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)


def get_object(full_class_string, **kwargs):
    """Create an object from the specified class..

    Parameters
    ----------
    full_class_string : str
        The full class string to the given python class.
        Example:
            my_project.my_module.my_class
    **kwargs
        keyword arguments that will be passed on to __init__ method.

    Returns
    -------
    object
        Returns the intilialized object of the specified class.
    """
    return load_class(full_class_string)(**kwargs)

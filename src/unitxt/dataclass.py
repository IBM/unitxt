import copy
import dataclasses
import functools
import inspect
from abc import ABCMeta
from inspect import Parameter, Signature
from typing import Any, Dict, List, Optional, final

_FIELDS = "__fields__"


class Undefined:
    pass


@dataclasses.dataclass
class Field:
    """An alternative to dataclasses.dataclass decorator for a more flexible field definition.

    Args:
        default (Any, optional):
            Default value for the field. Defaults to None.
        name (str, optional):
            Name of the field. Defaults to None.
        type (type, optional):
            Type of the field. Defaults to None.
        default_factory (Any, optional):
            A function that returns the default value. Defaults to None.
        final (bool, optional):
            A boolean indicating if the field is final (cannot be overridden). Defaults to False.
        abstract (bool, optional):
            A boolean indicating if the field is abstract (must be implemented by subclasses). Defaults to False.
        required (bool, optional):
            A boolean indicating if the field is required. Defaults to False.
        origin_cls (type, optional):
            The original class that defined the field. Defaults to None.
    """

    default: Any = Undefined
    name: str = None
    type: type = None
    init: bool = True
    also_positional: bool = True
    default_factory: Any = None
    final: bool = False
    abstract: bool = False
    required: bool = False
    internal: bool = False
    origin_cls: type = None
    metadata: Dict[str, str] = dataclasses.field(default_factory=dict)

    def get_default(self):
        if self.default_factory is not None:
            return self.default_factory()
        return self.default


@dataclasses.dataclass
class FinalField(Field):
    def __post_init__(self):
        self.final = True


@dataclasses.dataclass
class RequiredField(Field):
    def __post_init__(self):
        self.required = True


class MissingDefaultError(TypeError):
    pass


@dataclasses.dataclass
class OptionalField(Field):
    def __post_init__(self):
        self.required = False
        if self.default is Undefined and self.default_factory is None:
            raise MissingDefaultError(
                "OptionalField must have default or default_factory"
            )


@dataclasses.dataclass
class AbstractField(Field):
    def __post_init__(self):
        self.abstract = True


@dataclasses.dataclass
class NonPositionalField(Field):
    def __post_init__(self):
        self.also_positional = False


@dataclasses.dataclass
class InternalField(Field):
    def __post_init__(self):
        self.internal = True
        self.init = False
        self.also_positional = False


class FinalFieldError(TypeError):
    pass


class RequiredFieldError(TypeError):
    pass


class AbstractFieldError(TypeError):
    pass


class TypeMismatchError(TypeError):
    pass


class UnexpectedArgumentError(TypeError):
    pass


standard_variables = dir(object)


def is_class_method(func):
    if inspect.ismethod(func):
        return True
    if inspect.isfunction(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) > 0 and params[0].name in ["self", "cls"]:
            return True
    return False


def is_possible_field(field_name, field_value):
    """Check if a name-value pair can potentially represent a field.

    Args:
        field_name (str): The name of the field.
        field_value: The value of the field.

    Returns:
        bool: True if the name-value pair can represent a field, False otherwise.
    """
    if field_name in standard_variables:
        return False
    if is_class_method(field_value):
        return False
    return True


def get_fields(cls, attrs):
    """Get the fields for a class based on its attributes.

    Args:
        cls (type): The class to get the fields for.
        attrs (dict): The attributes of the class.

    Returns:
        dict: A dictionary mapping field names to Field instances.
    """
    fields = {}
    for base in cls.__bases__:
        fields = {**getattr(base, _FIELDS, {}), **fields}
    annotations = {**attrs.get("__annotations__", {})}

    for attr_name, attr_value in attrs.items():
        if attr_name not in annotations and is_possible_field(attr_name, attr_value):
            if attr_name in fields:
                try:
                    if not isinstance(attr_value, fields[attr_name].type):
                        raise TypeMismatchError(
                            f"Type mismatch for field '{attr_name}' of class '{fields[attr_name].origin_cls}'. Expected {fields[attr_name].type}, got {type(attr_value)}"
                        )
                except TypeError:
                    pass
                annotations[attr_name] = fields[attr_name].type

    for field_name, field_type in annotations.items():
        if field_name in fields and fields[field_name].final:
            raise FinalFieldError(
                f"Final field {field_name} defined in {fields[field_name].origin_cls} overridden in {cls}"
            )

        args = {
            "name": field_name,
            "type": field_type,
            "origin_cls": attrs["__qualname__"],
        }

        if field_name in attrs:
            field_value = attrs[field_name]
            if isinstance(field_value, Field):
                args = {**dataclasses.asdict(field_value), **args}
            elif isinstance(field_value, dataclasses.Field):
                args = {
                    "default": field_value.default,
                    "name": field_value.name,
                    "type": field_value.type,
                    "init": field_value.init,
                    "default_factory": field_value.default_factory,
                    **args,
                }
            else:
                args["default"] = field_value
                args["default_factory"] = None
        else:
            args["default"] = dataclasses.MISSING
            args["default_factory"] = None
            args["required"] = True

        field_instance = Field(**args)
        fields[field_name] = field_instance

        if cls.__allow_unexpected_arguments__:
            fields["_argv"] = InternalField(name="_argv", type=tuple, default=())
            fields["_kwargs"] = InternalField(name="_kwargs", type=dict, default={})

    return fields


def is_dataclass(obj):
    """Returns True if obj is a dataclass or an instance of a dataclass."""
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, _FIELDS)


def class_fields(obj):
    all_fields = fields(obj)
    return [
        field for field in all_fields if field.origin_cls == obj.__class__.__qualname__
    ]


def fields(cls):
    return list(getattr(cls, _FIELDS).values())


def fields_names(cls):
    return list(getattr(cls, _FIELDS).keys())


def external_fields_names(cls):
    return [field.name for field in fields(cls) if not field.internal]


def final_fields(cls):
    return [field for field in fields(cls) if field.final]


def required_fields(cls):
    return [field for field in fields(cls) if field.required]


def abstract_fields(cls):
    return [field for field in fields(cls) if field.abstract]


def is_abstract_field(field):
    return field.abstract


def is_final_field(field):
    return field.final


def get_field_default(field):
    if field.default_factory is not None:
        return field.default_factory()

    return field.default


def asdict(obj):
    assert is_dataclass(
        obj
    ), f"{obj} must be a dataclass, got {type(obj)} with bases {obj.__class__.__bases__}"
    return _asdict_inner(obj)


def _asdict_inner(obj):
    if is_dataclass(obj):
        return obj.to_dict()

    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # named tuple
        return type(obj)(*[_asdict_inner(v) for v in obj])

    if isinstance(obj, (list, tuple)):
        return type(obj)([_asdict_inner(v) for v in obj])

    if isinstance(obj, dict):
        return type(obj)({_asdict_inner(k): _asdict_inner(v) for k, v in obj.items()})

    return copy.deepcopy(obj)


def to_dict(obj, func=copy.deepcopy, _visited=None):
    """Recursively converts an object into a dictionary representation while avoiding infinite recursion due to circular references.

    Args:
        obj: Any Python object to be converted into a dictionary-like structure.
        func (Callable, optional): A function applied to non-iterable objects. Defaults to `copy.deepcopy`.
        _visited (set, optional): A set of object IDs used to track visited objects and prevent infinite recursion.

    Returns:
        dict: A dictionary representation of the input object, with supported collections and dataclasses
        recursively processed.

    Notes:
        - Supports dataclasses, named tuples, lists, tuples, and dictionaries.
        - Circular references are detected using object IDs and replaced by `func(obj)`.
        - Named tuples retain their original type instead of being converted to dictionaries.
    """
    # Initialize visited set on first call
    if _visited is None:
        _visited = set()

    # Get object ID to track visited objects
    obj_id = id(obj)

    # If we've seen this object before, return a placeholder to avoid infinite recursion
    if obj_id in _visited:
        return func(obj)

    # For mutable objects, add to visited set before recursing
    if (
        isinstance(obj, (dict, list))
        or is_dataclass(obj)
        or (isinstance(obj, tuple) and hasattr(obj, "_fields"))
    ):
        _visited.add(obj_id)

    if is_dataclass(obj):
        return {
            field.name: to_dict(getattr(obj, field.name), func, _visited)
            for field in fields(obj)
        }

    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # named tuple
        return type(obj)(*[to_dict(v, func, _visited) for v in obj])

    if isinstance(obj, (list, tuple)):
        return type(obj)([to_dict(v, func, _visited) for v in obj])

    if isinstance(obj, dict):
        return type(obj)(
            {
                to_dict(k, func, _visited): to_dict(v, func, _visited)
                for k, v in obj.items()
            }
        )

    return func(obj)


class DataclassMeta(ABCMeta):
    """Metaclass for Dataclass.

    Checks for final fields when a subclass is created.
    """

    @final
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        fields = get_fields(cls, attrs)
        setattr(cls, _FIELDS, fields)
        cls.update_init_signature()

    def update_init_signature(cls):
        parameters = []

        for name, field in getattr(cls, _FIELDS).items():
            if field.init and not field.internal:
                if field.default is not Undefined:
                    default_value = field.default
                elif field.default_factory is not None:
                    default_value = field.default_factory()
                else:
                    default_value = Parameter.empty

                if isinstance(default_value, dataclasses._MISSING_TYPE):
                    default_value = Parameter.empty
                param = Parameter(
                    name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=default_value,
                    annotation=field.type,
                )
                parameters.append(param)

        if getattr(cls, "__allow_unexpected_arguments__", False):
            parameters.append(Parameter("_argv", Parameter.VAR_POSITIONAL))
            parameters.append(Parameter("_kwargs", Parameter.VAR_KEYWORD))

        signature = Signature(parameters, __validate_parameters__=False)

        original_init = cls.__init__

        @functools.wraps(original_init)
        def custom_cls_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

        custom_cls_init.__signature__ = signature
        cls.__init__ = custom_cls_init


class Dataclass(metaclass=DataclassMeta):
    """Base class for data-like classes that provides additional functionality and control.

    Base class for data-like classes that provides additional functionality and control
    over Python's built-in @dataclasses.dataclass decorator. Other classes can inherit from
    this class to get the benefits of this implementation. As a base class, it ensures that
    all subclasses will automatically be data classes.

    The usage and field definitions are similar to Python's built-in @dataclasses.dataclass decorator.
    However, this implementation provides additional classes for defining "final", "required",
    and "abstract" fields.

    Key enhancements of this custom implementation:

    1. Automatic Data Class Creation: All subclasses automatically become data classes,
       without needing to use the @dataclasses.dataclass decorator.

    2. Field Immutability: Supports creation of "final" fields (using FinalField class) that
       cannot be overridden by subclasses. This functionality is not natively supported in
       Python or in the built-in dataclasses module.

    3. Required Fields: Supports creation of "required" fields (using RequiredField class) that
       must be provided when creating an instance of the class, adding a level of validation
       not present in the built-in dataclasses module.

    4. Abstract Fields: Supports creation of "abstract" fields (using AbstractField class) that
       must be overridden by any non-abstract subclass. This is similar to abstract methods in
       an abc.ABC class, but applied to fields.

    5. Type Checking: Performs type checking to ensure that if a field is redefined in a subclass,
       the type of the field remains consistent, adding static type checking not natively supported
       in Python.

    6. Error Definitions: Defines specific error types (FinalFieldError, RequiredFieldError,
       AbstractFieldError, TypeMismatchError) for providing detailed error information during debugging.

    7. MetaClass Usage: Uses a metaclass (DataclassMeta) for customization of class creation,
       allowing checks and alterations to be made at the time of class creation, providing more control.

    :Example:

    .. code-block:: python

        class Parent(Dataclass):
            final_field: int = FinalField(1)  # this field cannot be overridden
            required_field: str = RequiredField()
            also_required_field: float
            abstract_field: int = AbstractField()

        class Child(Parent):
            abstract_field = 3  # now once overridden, this is no longer abstract
            required_field = Field(name="required_field", default="provided", type=str)

        class Mixin(Dataclass):
            mixin_field = Field(name="mixin_field", default="mixin", type=str)

        class GrandChild(Child, Mixin):
            pass

        grand_child = GrandChild()
        logger.info(grand_child.to_dict())

        ...
    """

    __allow_unexpected_arguments__ = False

    @final
    def __init__(self, *argv, **kwargs):
        """Initialize fields based on kwargs.

        Checks for abstract fields when an instance is created.
        """
        super().__init__()
        _init_fields = [field for field in fields(self) if field.init]
        _init_fields_names = [field.name for field in _init_fields]
        _init_positional_fields_names = [
            field.name for field in _init_fields if field.also_positional
        ]

        for name in _init_positional_fields_names[: len(argv)]:
            if name in kwargs:
                raise TypeError(
                    f"{self.__class__.__name__} got multiple values for argument '{name}'"
                )

        expected_unexpected_argv = kwargs.pop("_argv", None)

        if len(argv) <= len(_init_positional_fields_names):
            unexpected_argv = []
        else:
            unexpected_argv = argv[len(_init_positional_fields_names) :]

        if expected_unexpected_argv is not None:
            assert (
                len(unexpected_argv) == 0
            ), f"Cannot specify both _argv and unexpected positional arguments. Got {unexpected_argv}"
            unexpected_argv = tuple(expected_unexpected_argv)

        expected_unexpected_kwargs = kwargs.pop("_kwargs", None)
        unexpected_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in _init_fields_names and k not in ["_argv", "_kwargs"]
        }

        if expected_unexpected_kwargs is not None:
            intersection = set(unexpected_kwargs.keys()) & set(
                expected_unexpected_kwargs.keys()
            )
            assert (
                len(intersection) == 0
            ), f"Cannot specify the same arguments in both _kwargs and in unexpected keyword arguments. Got {intersection} in both."
            unexpected_kwargs = {**unexpected_kwargs, **expected_unexpected_kwargs}

        if self.__allow_unexpected_arguments__:
            if len(unexpected_argv) > 0:
                kwargs["_argv"] = unexpected_argv
            if len(unexpected_kwargs) > 0:
                kwargs["_kwargs"] = unexpected_kwargs

        else:
            if len(unexpected_argv) > 0:
                raise UnexpectedArgumentError(
                    f"Too many positional arguments {unexpected_argv} for class {self.__class__.__name__}.\nShould be only {len(_init_positional_fields_names)} positional arguments: {', '.join(_init_positional_fields_names)}"
                )

            if len(unexpected_kwargs) > 0:
                raise UnexpectedArgumentError(
                    f"Unexpected keyword argument(s) {unexpected_kwargs} for class {self.__class__.__name__}.\nShould be one of: {external_fields_names(self)}"
                )

        for name, arg in zip(_init_positional_fields_names, argv):
            kwargs[name] = arg

        for field in abstract_fields(self):
            raise AbstractFieldError(
                f"Abstract field '{field.name}' of class {field.origin_cls} not implemented in {self.__class__.__name__}"
            )

        for field in required_fields(self):
            if field.name not in kwargs:
                raise RequiredFieldError(
                    f"Required field '{field.name}' of class {field.origin_cls} not set in {self.__class__.__name__}"
                )

        self.__pre_init__(**kwargs)

        for field in fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])
            else:
                setattr(self, field.name, get_field_default(field))

        self.__post_init__()

    @property
    def __is_dataclass__(self) -> bool:
        return True

    def __pre_init__(self, **kwargs):
        """Pre initialization hook."""
        pass

    def __post_init__(self):
        """Post initialization hook."""
        pass

    def _to_raw_dict(self):
        """Convert to raw dict."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def to_dict(self, classes: Optional[List] = None, keep_empty: bool = True):
        """Convert to dict.

        Args:
            classes (List, optional): List of parent classes which attributes should
                be returned. If set to None, then all class' attributes are returned.
            keep_empty (bool): If True, then  parameters are returned regardless if
                their values are None or not.
        """
        if not classes:
            attributes_dict = _asdict_inner(self._to_raw_dict())
        else:
            attributes = []
            for cls in classes:
                attributes += list(cls.__annotations__.keys())
            attributes_dict = {
                attribute: getattr(self, attribute) for attribute in attributes
            }

        return {
            attribute: value
            for attribute, value in attributes_dict.items()
            if keep_empty or value is not None
        }

    def get_repr_dict(self):
        result = {}
        for field in fields(self):
            if not field.internal:
                result[field.name] = getattr(self, field.name)
        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({', '.join([f'{key}={val!r}' for key, val in self.get_repr_dict().items()])})"

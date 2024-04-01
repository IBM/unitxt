import copy
import dataclasses
import functools
import warnings
from abc import ABCMeta
from inspect import Parameter, Signature
from typing import Any, Dict, final

_FIELDS = "__fields__"


class Undefined:
    pass


@dataclasses.dataclass
class Field:
    """An alternative to dataclasses.dataclass decorator for a more flexible field definition.

    Attributes:
        default (Any, optional): Default value for the field. Defaults to None.
        name (str, optional): Name of the field. Defaults to None.
        type (type, optional): Type of the field. Defaults to None.
        default_factory (Any, optional): A function that returns the default value. Defaults to None.
        final (bool, optional): A boolean indicating if the field is final (cannot be overridden). Defaults to False.
        abstract (bool, optional): A boolean indicating if the field is abstract (must be implemented by subclasses). Defaults to False.
        required (bool, optional): A boolean indicating if the field is required. Defaults to False.
        origin_cls (type, optional): The original class that defined the field. Defaults to None.
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
    deprecated: bool = False
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
class DeprecatedField(Field):
    def __post_init__(self):
        self.deprecated = True


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


def is_possible_field(field_name, field_value):
    """Check if a name-value pair can potentially represent a field.

    Args:
        field_name (str): The name of the field.
        field_value: The value of the field.

    Returns:
        bool: True if the name-value pair can represent a field, False otherwise.
    """
    return (
        field_name not in standard_variables
        and not field_name.startswith("__")
        and not callable(field_value)
    )


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
            field = attrs[field_name]
            if isinstance(field, Field):
                args = {**dataclasses.asdict(field), **args}
            elif isinstance(field, dataclasses.Field):
                args = {
                    "default": field.default,
                    "name": field.name,
                    "type": field.type,
                    "init": field.init,
                    "default_factory": field.default_factory,
                    **args,
                }
            else:
                args["default"] = field
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


def final_fields(cls):
    return [field for field in fields(cls) if field.final]


def required_fields(cls):
    return [field for field in fields(cls) if field.required]


def deprecated_fields(cls):
    return [field for field in fields(cls) if field.deprecated]


def abstract_fields(cls):
    return [field for field in fields(cls) if field.abstract]


def is_abstract_field(field):
    return field.abstract


def is_final_field(field):
    return field.final


def is_deprecated_field(field):
    return field.deprecated


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

    Example:
    .. highlight:: python
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
        Warn when a deprecated is used
        """
        _init_fields = [field for field in fields(self) if field.init]
        _init_fields_names = [field.name for field in _init_fields]
        _init_positional_fields_names = [
            field.name for field in _init_fields if field.also_positional
        ]

        _init_deprecated_fields = [field for field in _init_fields if field.deprecated]
        for dep_field in _init_deprecated_fields:
            warnings.warn(
                dep_field.metadata["deprecation_msg"], DeprecationWarning, stacklevel=2
            )

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
                    f"Unexpected keyword argument(s) {unexpected_kwargs} for class {self.__class__.__name__}.\nShould be one of: {fields_names(self)}"
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

    def to_dict(self):
        """Convert to dict."""
        return _asdict_inner(self._to_raw_dict())

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({', '.join([f'{field.name}={getattr(self, field.name)!r}' for field in fields(self)])})"

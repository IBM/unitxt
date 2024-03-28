import difflib
import inspect
import json
import os
import pkgutil
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Union, final

from .dataclass import AbstractField, Dataclass, Field, InternalField, fields
from .logging_utils import get_logger
from .parsing_utils import (
    separate_inside_and_outside_square_brackets,
)
from .settings_utils import get_settings
from .text_utils import camel_to_snake_case, is_camel_case
from .type_utils import issubtype
from .utils import artifacts_json_cache, save_json

logger = get_logger()
settings = get_settings()


class Artifactories:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.instance.artifactories = []

        return cls.instance

    def __iter__(self):
        self._index = 0  # Initialize/reset the index for iteration
        return self

    def __next__(self):
        while self._index < len(self.artifactories):
            artifactory = self.artifactories[self._index]
            self._index += 1
            if (
                settings.use_only_local_catalogs and not artifactory.is_local
            ):  # Corrected typo from 'is_loacl' to 'is_local'
                continue
            return artifactory
        raise StopIteration

    def register(self, artifactory):
        assert isinstance(
            artifactory, Artifactory
        ), "Artifactory must be an instance of Artifactory"
        assert hasattr(
            artifactory, "__contains__"
        ), "Artifactory must have __contains__ method"
        assert hasattr(
            artifactory, "__getitem__"
        ), "Artifactory must have __getitem__ method"
        self.artifactories = [artifactory, *self.artifactories]

    def unregister(self, artifactory):
        assert isinstance(
            artifactory, Artifactory
        ), "Artifactory must be an instance of Artifactory"
        assert hasattr(
            artifactory, "__contains__"
        ), "Artifactory must have __contains__ method"
        assert hasattr(
            artifactory, "__getitem__"
        ), "Artifactory must have __getitem__ method"
        self.artifactories.remove(artifactory)

    def reset(self):
        self.artifactories = []


def map_values_in_place(object, mapper):
    if isinstance(object, dict):
        for key, value in object.items():
            object[key] = mapper(value)
        return object
    if isinstance(object, list):
        for i in range(len(object)):
            object[i] = mapper(object[i])
        return object
    return mapper(object)


def get_closest_artifact_type(type):
    artifact_type_options = list(Artifact._class_register.keys())
    matches = difflib.get_close_matches(type, artifact_type_options)
    if matches:
        return matches[0]  # Return the closest match
    return None


class UnrecognizedArtifactTypeError(ValueError):
    def __init__(self, type) -> None:
        maybe_class = "".join(word.capitalize() for word in type.split("_"))
        message = f"'{type}' is not a recognized artifact 'type'. Make sure a the class defined this type (Probably called '{maybe_class}' or similar) is defined and/or imported anywhere in the code executed."
        closest_artifact_type = get_closest_artifact_type(type)
        if closest_artifact_type is not None:
            message += "\n\n" f"Did you mean '{closest_artifact_type}'?"
        super().__init__(message)


class MissingArtifactTypeError(ValueError):
    def __init__(self, dic) -> None:
        message = (
            f"Missing 'type' parameter. Expected 'type' in artifact dict, got {dic}"
        )
        super().__init__(message)


class Artifact(Dataclass):
    _class_register = {}

    type: str = Field(default=None, final=True, init=False)
    __description__: str = InternalField(
        default=None, required=False, also_positional=False
    )
    __tags__: Dict[str, str] = InternalField(
        default_factory=dict, required=False, also_positional=False
    )
    __id__: str = InternalField(default=None, required=False, also_positional=False)

    @classmethod
    def is_artifact_dict(cls, d):
        return isinstance(d, dict) and "type" in d

    @classmethod
    def verify_artifact_dict(cls, d):
        if not isinstance(d, dict):
            raise ValueError(
                f"Artifact dict <{d}> must be of type 'dict', got '{type(d)}'."
            )
        if "type" not in d:
            raise MissingArtifactTypeError(d)
        if not cls.is_registered_type(d["type"]):
            raise UnrecognizedArtifactTypeError(d["type"])

    @classmethod
    def get_artifact_type(cls):
        return camel_to_snake_case(cls.__name__)

    @classmethod
    def register_class(cls, artifact_class):
        assert issubclass(
            artifact_class, Artifact
        ), f"Artifact class must be a subclass of Artifact, got '{artifact_class}'"
        assert is_camel_case(
            artifact_class.__name__
        ), f"Artifact class name must be legal camel case, got '{artifact_class.__name__}'"

        snake_case_key = camel_to_snake_case(artifact_class.__name__)

        if cls.is_registered_type(snake_case_key):
            assert (
                str(cls._class_register[snake_case_key]) == str(artifact_class)
            ), f"Artifact class name must be unique, '{snake_case_key}' already exists for {cls._class_register[snake_case_key]}. Cannot be overridden by {artifact_class}."

            return snake_case_key

        cls._class_register[snake_case_key] = artifact_class

        return snake_case_key

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register_class(cls)

    @classmethod
    def is_artifact_file(cls, path):
        if not os.path.exists(path) or not os.path.isfile(path):
            return False
        with open(path) as f:
            d = json.load(f)
        return cls.is_artifact_dict(d)

    @classmethod
    def is_registered_type(cls, type: str):
        return type in cls._class_register

    @classmethod
    def is_registered_class_name(cls, class_name: str):
        snake_case_key = camel_to_snake_case(class_name)
        return cls.is_registered_type(snake_case_key)

    @classmethod
    def is_registered_class(cls, clz: object):
        return clz in set(cls._class_register.values())

    @classmethod
    def _recursive_load(cls, obj):
        if isinstance(obj, dict):
            new_d = {}
            for key, value in obj.items():
                new_d[key] = cls._recursive_load(value)
            obj = new_d
        elif isinstance(obj, list):
            obj = [cls._recursive_load(value) for value in obj]
        else:
            pass
        if cls.is_artifact_dict(obj):
            cls.verify_artifact_dict(obj)
            return cls._class_register[obj.pop("type")](**obj)

        return obj

    @classmethod
    def from_dict(cls, d, overwrite_args=None):
        if overwrite_args is not None:
            d = {**d, **overwrite_args}
        cls.verify_artifact_dict(d)
        return cls._recursive_load(d)

    @classmethod
    def load(cls, path, artifact_identifier=None, overwrite_args=None):
        d = artifacts_json_cache(path)
        new_artifact = cls.from_dict(d, overwrite_args=overwrite_args)
        new_artifact.__id__ = artifact_identifier
        return new_artifact

    def prepare(self):
        pass

    def verify(self):
        pass

    @final
    def __pre_init__(self, **kwargs):
        self._init_dict = get_raw(kwargs)

    @final
    def __post_init__(self):
        self.type = self.register_class(self.__class__)

        for field in fields(self):
            if issubtype(
                field.type, Union[Artifact, List[Artifact], Dict[str, Artifact]]
            ):
                value = getattr(self, field.name)
                value = map_values_in_place(value, maybe_recover_artifact)
                setattr(self, field.name, value)

        self.prepare()
        self.verify()

    def _to_raw_dict(self):
        return {"type": self.type, **self._init_dict}

    def save(self, path):
        data = self.to_dict()
        save_json(path, data)


def get_raw(obj):
    if isinstance(obj, Artifact):
        return obj._to_raw_dict()

    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # named tuple
        return type(obj)(*[get_raw(v) for v in obj])

    if isinstance(obj, (list, tuple)):
        return type(obj)([get_raw(v) for v in obj])

    if isinstance(obj, dict):
        return type(obj)({get_raw(k): get_raw(v) for k, v in obj.items()})

    return deepcopy(obj)


class ArtifactList(list, Artifact):
    def prepare(self):
        for artifact in self:
            artifact.prepare()


class Artifactory(Artifact):
    is_local: bool = AbstractField()

    @abstractmethod
    def __contains__(self, name: str) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, name) -> Artifact:
        pass

    @abstractmethod
    def get_with_overwrite(self, name, overwrite_args) -> Artifact:
        pass


class UnitxtArtifactNotFoundError(Exception):
    def __init__(self, name, artifactories):
        self.name = name
        self.artifactories = artifactories

    def __str__(self):
        msg = f"Artifact {self.name} does not exist, in artifactories:{self.artifactories}."
        if settings.use_only_local_catalogs:
            msg += f" Notice that unitxt.settings.use_only_local_catalogs is set to True, if you want to use remote catalogs set this settings or the environment variable {settings.use_only_local_catalogs_key}."
        return f"Artifact {self.name} does not exist, in artifactories:{self.artifactories}"


def fetch_artifact(name):
    if Artifact.is_artifact_file(name):
        return Artifact.load(name), None

    artifactory, name, args = get_artifactory_name_and_args(name=name)

    return artifactory.get_with_overwrite(name, overwrite_args=args), artifactory


def get_artifactory_name_and_args(
    name: str, artifactories: Optional[List[Artifactory]] = None
):
    name, args = separate_inside_and_outside_square_brackets(name)

    if artifactories is None:
        artifactories = list(Artifactories())

    for artifactory in artifactories:
        if name in artifactory:
            return artifactory, name, args

    raise UnitxtArtifactNotFoundError(name, artifactories)


def verbosed_fetch_artifact(identifier):
    artifact, artifactory = fetch_artifact(identifier)
    logger.info(f"Artifact {identifier} is fetched from {artifactory}")
    return artifact


def reset_artifacts_json_cache():
    artifacts_json_cache.cache_clear()


def maybe_recover_artifact(artifact):
    if isinstance(artifact, str):
        return verbosed_fetch_artifact(artifact)

    return artifact


def register_all_artifacts(path):
    for loader, module_name, _is_pkg in pkgutil.walk_packages(path):
        logger.info(__name__)
        if module_name == __name__:
            continue
        logger.info(f"Loading {module_name}")
        # Import the module
        module = loader.find_module(module_name).load_module(module_name)

        # Iterate over every object in the module
        for _name, obj in inspect.getmembers(module):
            # Make sure the object is a class
            if inspect.isclass(obj):
                # Make sure the class is a subclass of Artifact (but not Artifact itself)
                if issubclass(obj, Artifact) and obj is not Artifact:
                    logger.info(obj)

import difflib
import inspect
import json
import os
import pkgutil
from abc import abstractmethod
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Union, final

from .dataclass import Dataclass, Field, InternalField, fields
from .logging_utils import get_logger
from .text_utils import camel_to_snake_case, is_camel_case
from .type_utils import issubtype
from .utils import load_json, save_json

logger = get_logger()


class Artifactories:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.instance.artifactories = []

        return cls.instance

    def __iter__(self):
        return iter(self.artifactories)

    def __next__(self):
        return next(self.artifactories)

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
    type: str = Field(default=None, final=True, init=False)

    _class_register = {}

    artifact_identifier: str = InternalField(default=None, required=False)

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
            ), f"Artifact class name must be unique, '{snake_case_key}' already exists for {cls._class_register[snake_case_key]}. Cannot be overriden by {artifact_class}."

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
    def _recursive_load(cls, d):
        if isinstance(d, dict):
            new_d = {}
            for key, value in d.items():
                new_d[key] = cls._recursive_load(value)
            d = new_d
        elif isinstance(d, list):
            d = [cls._recursive_load(value) for value in d]
        else:
            pass
        if cls.is_artifact_dict(d):
            cls.verify_artifact_dict(d)
            return cls._class_register[d.pop("type")](**d)

        return d

    @classmethod
    def from_dict(cls, d):
        cls.verify_artifact_dict(d)
        return cls._recursive_load(d)

    @classmethod
    def load(cls, path, artifact_identifier=None):
        d = load_json(path)
        new_artifact = cls.from_dict(d)
        new_artifact.artifact_identifier = artifact_identifier
        return new_artifact

    def prepare(self):
        pass

    def verify(self):
        pass

    @final
    def __pre_init__(self, **kwargs):
        self._init_dict = deepcopy(kwargs)

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


class ArtifactList(list, Artifact):
    def prepare(self):
        for artifact in self:
            artifact.prepare()


class Artifactory(Artifact):
    @abstractmethod
    def __contains__(self, name: str) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, name) -> Artifact:
        pass


class UnitxtArtifactNotFoundError(Exception):
    def __init__(self, name, artifactories):
        self.name = name
        self.artifactories = artifactories

    def __str__(self):
        return f"Artifact {self.name} does not exist, in artifactories:{self.artifactories}"


@lru_cache(maxsize=None)
def fetch_artifact(name):
    if Artifact.is_artifact_file(name):
        return Artifact.load(name), None

    for artifactory in Artifactories():
        if name in artifactory:
            return artifactory[name], artifactory

    raise UnitxtArtifactNotFoundError(name, Artifactories().artifactories)


@lru_cache(maxsize=None)
def verbosed_fetch_artifact(identifer):
    artifact, artifactory = fetch_artifact(identifer)
    logger.info(f"Artifact {identifer} is fetched from {artifactory}")
    return artifact


def reset_artifacts_cache():
    fetch_artifact.cache_clear()
    verbosed_fetch_artifact.cache_clear()


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

import inspect
import json
import os
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import final, Any, Dict, List, Union
from .type_utils import issubtype
from .text_utils import camel_to_snake_case, is_camel_case


class AbstractField:
    pass


class Artifactories(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Artifactories, cls).__new__(cls)
            cls.instance.artifactories = []

        return cls.instance

    def __iter__(self):
        return iter(self.artifactories)

    def __next__(self):
        return next(self.artifactories)

    def register_atrifactory(self, artifactory):
        assert isinstance(artifactory, Artifactory), "Artifactory must be an instance of Artifactory"
        assert hasattr(artifactory, "__contains__"), "Artifactory must have __contains__ method"
        assert hasattr(artifactory, "__getitem__"), "Artifactory must have __getitem__ method"
        self.artifactories = [artifactory] + self.artifactories


class BaseArtifact(ABC):
    _class_register = {}

    @classmethod
    def is_artifact_dict(cls, d):
        return isinstance(d, dict) and "type" in d and d["type"] in cls._class_register

    @classmethod
    def register_class(cls, artifact_class):
        assert issubclass(artifact_class, BaseArtifact), "Artifact class must be a subclass of BaseArtifact"
        assert is_camel_case(
            artifact_class.__name__
        ), f"Artifact class name must be legal camel case, got {artifact_class.__name__}"

        snake_case_key = camel_to_snake_case(artifact_class.__name__)

        if snake_case_key in cls._class_register:
            assert (
                    cls._class_register[snake_case_key] == artifact_class
            ), f"Artifact class name must be unique, {snake_case_key} already exists for {cls._class_register[snake_case_key]}"

        cls._class_register[snake_case_key] = artifact_class

        return snake_case_key

    @classmethod
    def is_artifact_file(cls, path):
        if not os.path.exists(path) or not os.path.isfile(path):
            return False
        with open(path, "r") as f:
            d = json.load(f)
        return cls.is_artifact_dict(d)

    @final
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @final
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls = dataclass(cls)

    def prepare(self):
        pass

    def verify(self):
        pass

    @final
    def __post_init__(self):
        self.type = self.register_class(self.__class__)

        self._args_dict = asdict(self)

        for field in fields(self):
            if getattr(self, field.name) == 'cards.wnli':
                print('cards.wnli')
            if issubtype(field.type, Union[BaseArtifact, List[BaseArtifact], Dict[str, BaseArtifact]]):
                value = getattr(self, field.name)
                value = map_values_in_place(value, maybe_recover_artifact)
                setattr(self, field.name, value)

        self.prepare()
        self.verify()

    def to_dict(self):
        return self._args_dict

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    # def __getstate__(self):
    #     print('getstate', self.__dict__)
    #     return self.to_dict()

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
            instance = cls._class_register[d.pop("type")](**d)
            return instance
        else:
            return d

    @classmethod
    def from_dict(cls, d):
        assert cls.is_artifact_dict(d), "Input must be a dict with type field"
        return cls._recursive_load(d)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            d = json.load(f)

        assert "type" in d, "Saved artifact must have a type field"
        return cls._recursive_load(d)
        # assert d['type'] in cls._class_register, f'Artifact type "{d["type"]}" is not registered'
        # cls = cls._class_register[d.pop('type')]
        # return cls(**d)

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

class Artifact(BaseArtifact):
    type: str = field(init=False)


class ArtifactList(list, Artifact):
    def prepare(self):
        for artifact in self:
            artifact.prepare()


class Artifactory(Artifact, ABC):
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


def fetch_artifact(name):
    if Artifact.is_artifact_file(name):
        return Artifact.load(name), None
    else:
        for artifactory in Artifactories():
            if name in artifactory:
                return artifactory[name], artifactory

    raise UnitxtArtifactNotFoundError(name, Artifactories().artifactories)

def verbosed_fetch_artifact(identifer):
        artifact, artifactory = fetch_artifact(identifer)
        print(f"Artifact {identifer} is fetched from {artifactory}")
        return artifact

def maybe_recover_artifact(artifact):
    if isinstance(artifact, str):
        return verbosed_fetch_artifact(artifact)
    else:
        return artifact

def register_all_artifacts(path):
    for loader, module_name, is_pkg in pkgutil.walk_packages(path):
        print(__name__)
        if module_name == __name__:
            continue
        print(f"Loading {module_name}")
        # Import the module
        module = loader.find_module(module_name).load_module(module_name)

        # Iterate over every object in the module
        for name, obj in inspect.getmembers(module):
            # Make sure the object is a class
            if inspect.isclass(obj):
                # Make sure the class is a subclass of Artifact (but not Artifact itself)
                if issubclass(obj, Artifact) and obj is not Artifact:
                    print(obj)

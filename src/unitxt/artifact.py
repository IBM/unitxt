import difflib
import inspect
import json
import os
import pkgutil
import re
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, final

from .dataclass import (
    AbstractField,
    Dataclass,
    Field,
    InternalField,
    NonPositionalField,
    fields,
)
from .logging_utils import get_logger
from .parsing_utils import (
    separate_inside_and_outside_square_brackets,
)
from .settings_utils import get_constants, get_settings
from .text_utils import camel_to_snake_case, is_camel_case
from .type_utils import issubtype
from .utils import artifacts_json_cache, json_dump, save_to_file

logger = get_logger()
settings = get_settings()
constants = get_constants()


def is_name_legal_for_catalog(name):
    return re.match(r"^[\w" + constants.catalog_hierarchy_sep + "]+$", name)


def verify_legal_catalog_name(name):
    assert is_name_legal_for_catalog(
        name
    ), f'Artifict name ("{name}") should be alphanumeric. Use "." for nesting (e.g. myfolder.my_artifact)'


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

    __type__: str = Field(default=None, final=True, init=False)
    __description__: str = NonPositionalField(
        default=None, required=False, also_positional=False
    )
    __tags__: Dict[str, str] = NonPositionalField(
        default_factory=dict, required=False, also_positional=False
    )
    __id__: str = InternalField(default=None, required=False, also_positional=False)

    data_classification_policy: List[str] = NonPositionalField(
        default=None, required=False, also_positional=False
    )

    @classmethod
    def is_artifact_dict(cls, d):
        return isinstance(d, dict) and "__type__" in d

    @classmethod
    def verify_artifact_dict(cls, d):
        if not isinstance(d, dict):
            raise ValueError(
                f"Artifact dict <{d}> must be of type 'dict', got '{type(d)}'."
            )
        if "__type__" not in d:
            raise MissingArtifactTypeError(d)
        if not cls.is_registered_type(d["__type__"]):
            raise UnrecognizedArtifactTypeError(d["__type__"])

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
            return cls._class_register[obj.pop("__type__")](**obj)

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

    def get_pretty_print_name(self):
        if self.__id__ is not None:
            return self.__id__
        return self.__class__.__name__

    def prepare(self):
        pass

    def verify(self):
        pass

    @final
    def __pre_init__(self, **kwargs):
        self._init_dict = get_raw(kwargs)

    @final
    def verify_data_classification_policy(self):
        if self.data_classification_policy is not None:
            if not isinstance(self.data_classification_policy, list) or not all(
                isinstance(data_classification, str)
                for data_classification in self.data_classification_policy
            ):
                raise ValueError(
                    f"The 'data_classification_policy' of {self.get_pretty_print_name()} "
                    f"must be either None - in case when no policy applies - or a list of "
                    f"strings, for example: ['public']. However, '{self.data_classification_policy}' "
                    f"of type {type(self.data_classification_policy)} was provided instead."
                )

    @final
    def __post_init__(self):
        self.__type__ = self.register_class(self.__class__)

        for field in fields(self):
            if issubtype(
                field.type, Union[Artifact, List[Artifact], Dict[str, Artifact]]
            ):
                value = getattr(self, field.name)
                value = map_values_in_place(value, maybe_recover_artifact)
                setattr(self, field.name, value)

        self.verify_data_classification_policy()
        if not settings.skip_artifacts_prepare_and_verify:
            self.prepare()
            self.verify()

    def _to_raw_dict(self):
        return {"__type__": self.__type__, **self._init_dict}

    def to_json(self):
        data = self.to_dict()
        return json_dump(data)

    def serialize(self):
        if self.__id__ is not None:
            return self.__id__
        return self.to_json()

    def save(self, path):
        save_to_file(path, self.to_json())

    @classmethod
    def deserialize(cls, artifact_rep):
        data = json.loads(artifact_rep)
        return Artifact.from_dict(data)

    def verify_instance(
        self, instance: Dict[str, Any], name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Checks if data classifications of an artifact and instance are compatible.

        Raises an error if an artifact's data classification policy does not include that of
        processed data. The purpose is to ensure that any sensitive data is handled in a
        proper way (for example when sending it to some external services).

        Args:
            instance (Dict[str, Any]): data which should contain its allowed data
                classification policies under key 'data_classification_policy'.
            name (Optional[str]): name of artifact which should be used to retrieve
                data classification from env. If not specified, then either __id__ or
                 __class__.__name__, are used instead, respectively.

        Returns:
            Dict[str, Any]: unchanged instance.

        Examples:
            instance = {"x": "some_text", "data_classification_policy": ["pii"]}

            # Will raise an error as "pii" is not included policy
            metric = Accuracy(data_classification_policy=["public"])
            metric.verify_instance(instance)

            # Will not raise an error
            template = SpanLabelingTemplate(data_classification_policy=["pii", "propriety"])
            template.verify_instance(instance)

            # Will not raise an error since the policy was specified in environment variable:
            UNITXT_DATA_CLASSIFICATION_POLICY = json.dumps({"metrics.accuracy": ["pii"]})
            metric = fetch_artifact("metrics.accuracy")
            metric.verify_instance(instance)
        """
        name = name or self.get_pretty_print_name()
        data_classification_policy = get_artifacts_data_classification(name)
        if not data_classification_policy:
            data_classification_policy = self.data_classification_policy

        if not data_classification_policy:
            return instance

        instance_data_classification = instance.get("data_classification_policy")
        if not instance_data_classification:
            get_logger().warning(
                f"The data does not provide information if it can be used by "
                f"'{name}' with the following data classification policy "
                f"'{data_classification_policy}'. This may lead to sending of undesired "
                f"data to external service. Set the 'data_classification_policy' "
                f"of the data to ensure a proper handling of sensitive information."
            )
            return instance

        if not any(
            data_classification in data_classification_policy
            for data_classification in instance_data_classification
        ):
            raise ValueError(
                f"The instance '{instance} 'has the following data classification policy "
                f"'{instance_data_classification}', however, the artifact '{name}' "
                f"is only configured to support the data with classification "
                f"'{data_classification_policy}'. To enable this either change "
                f"the 'data_classification_policy' attribute of the artifact, "
                f"or modify the environment variable "
                f"'UNITXT_DATA_CLASSIFICATION_POLICY' accordingly."
            )

        return instance


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


def fetch_artifact(artifact_rep) -> Tuple[Artifact, Union[Artifactory, None]]:
    if isinstance(artifact_rep, Artifact):
        return artifact_rep, None
    if Artifact.is_artifact_file(artifact_rep):
        return Artifact.load(artifact_rep), None

    name, _ = separate_inside_and_outside_square_brackets(artifact_rep)
    if is_name_legal_for_catalog(name):
        artifactory, artifact_rep, args = get_artifactory_name_and_args(
            name=artifact_rep
        )
        return artifactory.get_with_overwrite(
            artifact_rep, overwrite_args=args
        ), artifactory

    return Artifact.deserialize(artifact_rep), None


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
    logger.debug(f"Artifact {identifier} is fetched from {artifactory}")
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


def get_artifacts_data_classification(artifact: str) -> Optional[List[str]]:
    """Loads given artifact's data classification policy from an environment variable.

    Args:
        artifact (str): Name of the artifact which the data classification policy
            should be retrieved for. For example "metrics.accuracy".

    Returns:
        Optional[List[str]] - Data classification policies for the specified artifact
            if they were found, or None otherwise.
    """
    data_classification = settings.data_classification_policy
    if data_classification is None:
        return None

    error_msg = (
        f"If specified, the value of 'UNITXT_DATA_CLASSIFICATION_POLICY' "
        f"should be a valid json dictionary. Got '{data_classification}' "
        f"instead."
    )

    try:
        data_classification = json.loads(data_classification)
    except json.decoder.JSONDecodeError as e:
        raise RuntimeError(error_msg) from e

    if not isinstance(data_classification, dict):
        raise RuntimeError(error_msg)

    for artifact_name, artifact_data_classifications in data_classification.items():
        if (
            not isinstance(artifact_name, str)
            or not isinstance(artifact_data_classifications, list)
            or not all(
                isinstance(artifact_data_classification, str)
                for artifact_data_classification in artifact_data_classifications
            )
        ):
            raise RuntimeError(
                "'UNITXT_DATA_CLASSIFICATION_POLICY' should be of type "
                "'Dict[str, List[str]]', where a artifact's name is a key, and a "
                "value is a list of data classifications used by that artifact."
            )

    if artifact not in data_classification.keys():
        return None

    return data_classification.get(artifact)

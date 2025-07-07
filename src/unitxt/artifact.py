import importlib
import inspect
import json
import os
import re
import sys
import sysconfig
import warnings
from abc import abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, final

from .dataclass import (
    AbstractField,
    Dataclass,
    Field,
    InternalField,
    NonPositionalField,
    fields,
)
from .error_utils import Documentation, UnitxtError, UnitxtWarning, error_context
from .logging_utils import get_logger
from .parsing_utils import (
    separate_inside_and_outside_square_brackets,
)
from .settings_utils import get_constants, get_settings
from .type_utils import isoftype, issubtype
from .utils import (
    artifacts_json_cache,
    json_dump,
    save_to_file,
    shallow_copy,
)

logger = get_logger()
settings = get_settings()
constants = get_constants()


@lru_cache(maxsize=1)
def _get_stdlib_path():
    return sysconfig.get_path("stdlib")


@lru_cache(maxsize=1)
def _get_site_packages_path():
    return sysconfig.get_path("purelib")


@lru_cache(maxsize=1)
def _get_stdlib_pattern():
    return re.compile(r"/lib/python\d+\.\d+/")


@lru_cache(maxsize=1)
def _get_all_site_packages_paths():
    paths = []
    # Get standard paths
    paths.append(sysconfig.get_path("purelib"))
    paths.append(sysconfig.get_path("platlib"))
    # Also check sys.path for additional site-packages and dist-packages
    for path in sys.path:
        if "site-packages" in path or "dist-packages" in path:
            paths.append(path)
    return list(set(paths))  # Remove duplicates


@lru_cache(maxsize=1)
def _get_site_packages_files():
    all_files = {}
    for site_packages in _get_all_site_packages_paths():
        if os.path.exists(site_packages):
            try:
                files = os.listdir(site_packages)
                all_files[site_packages] = frozenset(files)
            except (OSError, PermissionError):
                all_files[site_packages] = frozenset()
    return all_files


@lru_cache(maxsize=1)
def _get_editable_packages():
    editable_packages = set()
    all_site_packages_files = _get_site_packages_files()

    for _, files in all_site_packages_files.items():
        for filename in files:
            if filename.endswith(".egg-link"):
                # Extract package name from egg-link file
                package_name = filename[:-9]  # Remove .egg-link
                editable_packages.add(package_name)
            elif filename.endswith(".pth"):
                if filename.startswith("__editable__."):
                    # Modern pip editable installs: __editable__.package.pth
                    parts = filename.split(".")
                    if len(parts) >= 3:
                        package_name = parts[1]
                        editable_packages.add(package_name)
                # Also check for other .pth files that might contain package names
                # This mimics the original glob pattern *{package_name}*.pth behavior
                # but we'll check this during the main function call

    return frozenset(editable_packages)


# flake8: noqa: C901
@lru_cache(maxsize=512)
def is_library_module(module_name):
    r"""Determines if a given module is a library module (as opposed to a local/project module).

    A module is considered a library module if it falls into any of these categories:

    1. **Built-in modules**: Modules with no __file__ attribute or __file__ = None
       - Examples: sys, builtins, __main__

    2. **Standard library modules**: Modules that are part of Python's standard library
       - Direct path match: modules in sysconfig.get_path('stdlib')
       - Pattern match: modules in paths matching /lib/python\\d+\\.\\d+/ (but not in site-packages)
       - Examples: os, json, re, collections, urllib.parse

    3. **Installed packages**: Third-party packages installed via pip/conda
       - Modules in site-packages or dist-packages directories
       - Examples: requests, numpy, pandas

    4. **Editable installs**: Development packages installed with pip install -e
       - Modules outside site-packages but with corresponding installation files:
         - .egg-link files (older pip versions)
         - .pth files (various installation methods)
         - __editable__.{package}.pth files (modern pip versions)
       - Examples: local packages installed in development mode

    Returns False for:
    - **Local/project modules**: Modules that are part of the current project but not installed
    - **Non-existent modules**: Modules that cannot be imported
    - **Invalid input**: Empty strings, None, or other invalid module names

    Args:
        module_name (str): The name of the module to check (e.g., 'os', 'requests.api')

    Returns:
        bool: True if the module is a library module, False otherwise

    Raises:
        ValueError: If module_name is an empty string
        TypeError: If module_name is None or not a string

    Examples:
        >>> is_library_module('os')           # Standard library
        True
        >>> is_library_module('requests')     # Installed package
        True
        >>> is_library_module('my_project')   # Local module
        False
        >>> is_library_module('unitxt')       # Editable install
        True
    """
    if (
        module_name is None
        or (not isinstance(module_name, str))
        or len(module_name) == 0
    ):
        return False

    """Determines if a given module is a library module (as opposed to a local/project module).
    Fully cached version that minimizes all OS operations.
    """
    if not module_name or not isinstance(module_name, str):
        return False

    if module_name not in sys.modules:
        try:
            __import__(module_name)
        except ImportError:
            return False

    module = sys.modules[module_name]

    # Built-in modules
    if not hasattr(module, "__file__") or module.__file__ is None:
        return True

    file_path = module.__file__

    # Check for standard library (cached path)
    stdlib_path = _get_stdlib_path()
    if file_path.startswith(stdlib_path):
        return True

    # Check stdlib pattern (cached regex)
    stdlib_pattern = _get_stdlib_pattern()
    if stdlib_pattern.search(file_path) and "site-packages" not in file_path:
        return True

    # Check if it's in site-packages
    if any(pkg_dir in file_path for pkg_dir in ["site-packages", "dist-packages"]):
        return True

    # Check for editable installs (cached set + additional .pth file check)
    package_name = module_name.split(".")[0]
    editable_packages = _get_editable_packages()
    if package_name in editable_packages:
        return True

    # Additional check for .pth files containing package name (mimics original glob behavior)
    all_site_packages_files = _get_site_packages_files()
    for _, files in all_site_packages_files.items():
        for filename in files:
            if filename.endswith(".pth") and package_name in filename:
                return True

    return False


def import_module_from_file(file_path):
    # Get the module name (file name without extension)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    # Create a module specification
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # Create a new module based on the specification
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# type is the dict read from a catelog entry, the value of a key "__type__"
def get_module_class_names(artifact_type: dict):
    return artifact_type["module"], artifact_type["name"]


# type is the dict read from a catelog entry, the value of a key "__type__"
def get_class_from_artifact_type(type: dict):
    module_path, class_name = get_module_class_names(type)
    if module_path == "class_register":
        if class_name not in Artifact._class_register:
            raise ValueError(
                f"Can not instantiate a class from type {type}, because {class_name} is currently not registered in Artifact._class_register."
            )
        return Artifact._class_register[class_name]

    module = importlib.import_module(module_path)

    if "." not in class_name:
        if hasattr(module, class_name) and inspect.isclass(getattr(module, class_name)):
            return getattr(module, class_name)
        if class_name in Artifact._class_register:
            return Artifact._class_register[class_name]
        module_file = module.__file__ if hasattr(module, "__file__") else None
        if module_file:
            module = import_module_from_file(module_file)

        assert class_name in Artifact._class_register
        return Artifact._class_register[class_name]

    class_name_components = class_name.split(".")
    klass = getattr(module, class_name_components[0])
    for i in range(1, len(class_name_components)):
        klass = getattr(klass, class_name_components[i])
    return klass


def get_class_or_function_from_artifact_type(type: dict):
    module_path, class_name = get_module_class_names(type)
    module = importlib.import_module(module_path)

    if "." not in class_name:
        return getattr(module, class_name)

    class_name_components = class_name.split(".")
    klass = getattr(module, class_name_components[0])
    for i in range(1, len(class_name_components)):
        klass = getattr(klass, class_name_components[i])
    return klass


def is_artifact_dict(obj):
    return isinstance(obj, dict) and "__type__" in obj


def verify_artifact_dict(d):
    if not isinstance(d, dict):
        raise ValueError(
            f"Artifact dict <{d}> must be of type 'dict', got '{type(d)}'."
        )
    if "__type__" not in d:
        raise MissingArtifactTypeError(d)


def from_dict(d, overwrite_args=None):
    if overwrite_args is not None:
        d = {**d, **overwrite_args}
    verify_artifact_dict(d)
    return _recursive_load(d)


def _recursive_load(obj):
    if isinstance(obj, dict):
        obj = {key: _recursive_load(value) for key, value in obj.items()}
        if is_artifact_dict(obj):
            try:
                artifact_type = obj.pop("__type__")
                artifact_class = get_class_from_artifact_type(artifact_type)
                obj = artifact_class.process_data_after_load(obj)
                return artifact_class(**obj)
            except (ImportError, AttributeError) as e:
                raise UnrecognizedArtifactTypeError(artifact_type) from e
    elif isinstance(obj, list):
        return [_recursive_load(value) for value in obj]

    return obj


def is_name_legal_for_catalog(name):
    return re.match(r"^[\w" + constants.catalog_hierarchy_sep + "]+$", name)


def verify_legal_catalog_name(name):
    assert is_name_legal_for_catalog(
        name
    ), f'Artifict name ("{name}") should be alphanumeric. Use "." for nesting (e.g. myfolder.my_artifact)'


def dict_diff_string(dict1, dict2, max_diff=200):
    keys_in_both = dict1.keys() & dict2.keys()
    added = {k: dict2[k] for k in dict2.keys() - dict1.keys()}
    removed = {k: dict1[k] for k in dict1.keys() - dict2.keys()}
    changed = {}
    for k in keys_in_both:
        if str(dict1[k]) != str(dict2[k]):
            changed[k] = (dict1[k], dict2[k])
    result = []

    def format_with_value(k, value, label):
        value_str = str(value)
        return (
            f" - {k} ({label}): {value_str}"
            if len(value_str) <= max_diff
            else f" - {k} ({label})"
        )

    result.extend(format_with_value(k, added[k], "added") for k in added)
    result.extend(format_with_value(k, removed[k], "removed") for k in removed)
    result.extend(
        f" - {k} (changed): {dict1[k]!s} -> {dict2[k]!s}"
        if len(str(dict1[k])) <= max_diff and len(str(dict2[k])) <= 200
        else f" - {k} (changed)"
        for k in changed
    )

    return "\n".join(result)


class Catalogs:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.instance.catalogs = []

        return cls.instance

    def __iter__(self):
        self._index = 0  # Initialize/reset the index for iteration
        return self

    def __next__(self):
        while self._index < len(self.catalogs):
            catalog = self.catalogs[self._index]
            self._index += 1
            if (
                settings.use_only_local_catalogs and not catalog.is_local
            ):  # Corrected typo from 'is_loacl' to 'is_local'
                continue
            return catalog
        raise StopIteration

    def register(self, catalog):
        assert isinstance(
            catalog, AbstractCatalog
        ), "catalog must be an instance of AbstractCatalog"
        assert hasattr(catalog, "__contains__"), "catalog must have __contains__ method"
        assert hasattr(catalog, "__getitem__"), "catalog must have __getitem__ method"
        self.catalogs = [catalog, *self.catalogs]

    def unregister(self, catalog):
        assert isinstance(
            catalog, AbstractCatalog
        ), "catalog must be an instance of Catalog"
        assert hasattr(catalog, "__contains__"), "catalog must have __contains__ method"
        assert hasattr(catalog, "__getitem__"), "catalog must have __getitem__ method"
        self.catalogs.remove(catalog)

    def reset(self):
        self.catalogs = []


def maybe_recover_artifacts_structure(obj):
    if Artifact.is_possible_identifier(obj):
        return verbosed_fetch_artifact(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = maybe_recover_artifact(value)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = maybe_recover_artifact(obj[i])
        return obj
    return obj


class UnrecognizedArtifactTypeError(ValueError):
    def __init__(self, type) -> None:
        maybe_class = type["name"].split(".")[-1]
        message = f"'{type}' is not a recognized artifact 'type'. Make sure a class (Probably called '{maybe_class}' or similar) is defined and/or imported anywhere in the code executed."
        super().__init__(message)


class MissingArtifactTypeError(ValueError):
    def __init__(self, dic) -> None:
        message = (
            f"Missing '__type__' parameter. Expected 'type' in artifact dict, got {dic}"
        )
        super().__init__(message)


class Artifact(Dataclass):
    _class_register = {}

    __type__: dict = Field(default=None, final=True, init=False)
    __title__: str = NonPositionalField(
        default=None, required=False, also_positional=False
    )
    __description__: str = NonPositionalField(
        default=None, required=False, also_positional=False
    )
    __tags__: Dict[str, str] = NonPositionalField(
        default_factory=dict, required=False, also_positional=False
    )
    __id__: str = InternalField(default=None, required=False, also_positional=False)

    # if not None, the artifact is deprecated, and once instantiated, that msg
    # is logged as a warning
    __deprecated_msg__: str = NonPositionalField(
        default=None, required=False, also_positional=False
    )

    data_classification_policy: List[str] = NonPositionalField(
        default=None, required=False, also_positional=False
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        module = inspect.getmodule(cls)
        # standardize module name
        module_name = getattr(module, "__name__", None)
        if not is_library_module(module_name):
            cls.register_class()

    @classmethod
    def is_possible_identifier(cls, obj):
        return isinstance(obj, str) or is_artifact_dict(obj)

    @classmethod
    def get_artifact_type(cls):
        module = inspect.getmodule(cls)
        # standardize module name
        module_name = getattr(module, "__name__", None)
        if not is_library_module(module_name):
            non_library_module_warning = f"module named {module_name} is not importable. Class {cls} is thus registered into Artifact.class_register, indexed by {cls.__name__}, accessible there as long as this class_register lives."
            warnings.warn(non_library_module_warning, ImportWarning, stacklevel=2)
            cls.register_class()
            return {"module": "class_register", "name": cls.__name__}
        if hasattr(cls, "__qualname__") and "." in cls.__qualname__:
            return {"module": module_name, "name": cls.__qualname__}
        return {"module": module_name, "name": cls.__name__}

    @classmethod
    def register_class(cls):
        Artifact._class_register[cls.__name__] = cls

    @classmethod
    def is_artifact_file(cls, path):
        if not os.path.exists(path) or not os.path.isfile(path):
            return False
        with open(path) as f:
            d = json.load(f)
        return is_artifact_dict(d)

    @classmethod
    def load(cls, path, artifact_identifier=None, overwrite_args=None):
        d = artifacts_json_cache(path)
        if "__type__" in d and d["__type__"]["name"].endswith("ArtifactLink"):
            from_dict(d)  # for verifications and warnings
            catalog, artifact_rep, _ = get_catalog_name_and_args(name=d["to"])
            return catalog.get_with_overwrite(
                artifact_rep, overwrite_args=overwrite_args
            )

        new_artifact = from_dict(d, overwrite_args=overwrite_args)
        new_artifact.__id__ = artifact_identifier
        return new_artifact

    def get_pretty_print_name(self):
        if self.__id__ is not None:
            return self.__id__
        return self.__class__.__name__

    def prepare(self):
        if self.__deprecated_msg__:
            warnings.warn(self.__deprecated_msg__, DeprecationWarning, stacklevel=2)

    def prepare_args(self):
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
        # record module and class name as they are, without verifying instantiationability via python imports
        module = inspect.getmodule(self.__class__)
        # standardize module name
        module_name = getattr(module, "__name__", None)
        class_name = (
            self.__class__.__qualname__
            if hasattr(self.__class__, "__qualname__")
            and "." in self.__class__.__qualname__
            else self.__class__.__name__
        )
        self.__type__ = {"module": module_name, "name": class_name}
        ## now verify
        self.maybe_fix_type_to_ensure_instantiation_ability()

        for field in fields(self):
            if issubtype(
                field.type, Union[Artifact, List[Artifact], Dict[str, Artifact]]
            ):
                value = getattr(self, field.name)
                value = maybe_recover_artifacts_structure(value)
                setattr(self, field.name, value)

        self.verify_data_classification_policy()
        self.prepare_args()
        if not settings.skip_artifacts_prepare_and_verify:
            with error_context(self, action="Prepare Object"):
                self.prepare()
            with error_context(self, action="Verify Object"):
                self.verify()

    def _to_raw_dict(self):
        return {
            "__type__": self.__type__,
            **self.process_data_before_dump(self._init_dict),
        }

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        try:
            new_obj = from_dict(self.to_dict())
        except:
            # needed only for artifacts defined inline for testing etc. E.g. 'NERWithoutClassReporting'
            new_obj = self
        memo[id(self)] = new_obj
        return new_obj

    def process_data_before_dump(self, data):
        return data

    @classmethod
    def process_data_after_load(cls, data):
        return data

    def to_json(self):
        data = self.to_dict()

        return json_dump(data)

    def to_yaml(self):
        import yaml

        data = self.to_dict()
        return yaml.dump(data)

    def serialize(self):
        if self.__id__ is not None:
            return self.__id__
        return self.to_json()

    def maybe_fix_type_to_ensure_instantiation_ability(self):
        if (
            not is_library_module(self.__type__["module"])
            or "<locals>" in self.__type__["name"]
        ):
            self.__class__.register_class()
            self.__type__ = {
                "module": "class_register",
                "name": self.__class__.__name__,
            }
            return

    def save(self, path):
        original_args = from_dict(self.to_dict()).get_repr_dict()
        current_args = self.get_repr_dict()
        diffs = dict_diff_string(original_args, current_args)
        if diffs:
            raise UnitxtError(
                f"Cannot save catalog artifacts that have changed since initialization. Detected differences in the following fields:\n{diffs}"
            )
        save_to_file(path, self.to_json())

    def verify_instance(
        self, instance: Dict[str, Any], name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Checks if data classifications of an artifact and instance are compatible.

        Raises an error if an artifact's data classification policy does not include that of
        processed data. The purpose is to ensure that any sensitive data is handled in a
        proper way (for example when sending it to some external services).

        Args:
            instance (Dict[str, Any]): data which should contain its allowed data classification policies under key 'data_classification_policy'.

            name (Optional[str]): name of artifact which should be used to retrieve data classification from env. If not specified, then either ``__id__`` or ``__class__.__name__``, are used instead, respectively.

        Returns:
            Dict[str, Any]: unchanged instance.

        :Examples:

        .. code-block:: python

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

        if not isoftype(instance, Dict[str, Any]):
            raise ValueError(
                f"The instance passed to inference engine is not a dictionary. Instance:\n{instance}"
            )
        instance_data_classification = instance.get("data_classification_policy")
        if not instance_data_classification:
            UnitxtWarning(
                f"The data does not provide information if it can be used by "
                f"'{name}' with the following data classification policy "
                f"'{data_classification_policy}'. This may lead to sending of undesired "
                f"data to external service. Set the 'data_classification_policy' "
                f"of the data to ensure a proper handling of sensitive information.",
                Documentation.DATA_CLASSIFICATION_POLICY,
            )
            return instance

        with error_context(
            self,
            action="Sensitive Data Verification",
            help="https://www.unitxt.ai/en/latest/docs/data_classification_policy.html",
        ):
            if not any(
                data_classification in data_classification_policy
                for data_classification in instance_data_classification
            ):
                raise UnitxtError(
                    f"The instance '{instance} 'has the following data classification policy "
                    f"'{instance_data_classification}', however, the artifact '{name}' "
                    f"is only configured to support the data with classification "
                    f"'{data_classification_policy}'. To enable this either change "
                    f"the 'data_classification_policy' attribute of the artifact, "
                    f"or modify the environment variable "
                    f"'UNITXT_DATA_CLASSIFICATION_POLICY' accordingly.",
                    Documentation.DATA_CLASSIFICATION_POLICY,
                )

        return instance

    def __repr__(self):
        if self.__id__ is not None:
            return self.__id__
        return super().__repr__()


class ArtifactLink(Artifact):
    to: Artifact

    def verify(self):
        if self.to.__id__ is None:
            raise UnitxtError("ArtifactLink must link to existing catalog entry.")


def get_raw(obj):
    if isinstance(obj, Artifact):
        if obj.__id__ is not None:
            return obj.__id__
        return obj._to_raw_dict()

    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # named tuple
        return type(obj)(*[get_raw(v) for v in obj])

    if isinstance(obj, (list, tuple)):
        return type(obj)([get_raw(v) for v in obj])

    if isinstance(obj, dict):
        return type(obj)({get_raw(k): get_raw(v) for k, v in obj.items()})

    return shallow_copy(obj)


class ArtifactList(list, Artifact):
    def prepare(self):
        for artifact in self:
            artifact.prepare()


class AbstractCatalog(Artifact):
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


class UnitxtArtifactNotFoundError(UnitxtError):
    def __init__(self, name, catalogs):
        self.name = name
        self.catalogs = catalogs
        msg = (
            f"Artifact {self.name} does not exist, in Unitxt catalogs: {self.catalogs}."
        )
        if settings.use_only_local_catalogs:
            msg += f"\nNotice that unitxt.settings.use_only_local_catalogs is set to True, if you want to use remote catalogs set this settings or the environment variable {settings.use_only_local_catalogs_key}."
        super().__init__(msg)


def fetch_artifact(
    artifact_rep, overwrite_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[Artifact, Union[AbstractCatalog, None]]:
    """Loads an artifict from one of possible representations.

    (1) If artifact representation is already an Artifact object, return it.
    (2) If artifact representation is a string location of a local file, load the Artifact from the local file.
    (3) If artifact representation is a string name in the catalog, load the Artifact from the catalog.
    (4) If artifact representation is a json string, create a dictionary representation from the string and build an Artifact object from it.
    (5) Otherwise, check that the artifact representation is a dictionary and build an Artifact object from it.
    """
    if isinstance(artifact_rep, Artifact):
        if isinstance(artifact_rep, ArtifactLink):
            return fetch_artifact(artifact_rep.to)
        return artifact_rep, None

    # If local file
    if isinstance(artifact_rep, str) and Artifact.is_artifact_file(artifact_rep):
        artifact_to_return = Artifact.load(artifact_rep)

        return artifact_to_return, None

    # if artifact is a name of a catalog entry
    if isinstance(artifact_rep, str):
        name, _ = separate_inside_and_outside_square_brackets(artifact_rep)
        if is_name_legal_for_catalog(name):
            catalog, artifact_rep, args = get_catalog_name_and_args(name=artifact_rep)
            if overwrite_kwargs is not None:
                if args is None:
                    args = overwrite_kwargs
                else:
                    args.update(overwrite_kwargs)
            artifact_to_return = catalog.get_with_overwrite(
                artifact_rep, overwrite_args=args
            )
            return artifact_to_return, catalog

    # If Json string, first load into dictionary
    if isinstance(artifact_rep, str):
        artifact_rep = json.loads(artifact_rep)
    # Load from dictionary (fails if not valid dictionary)
    return from_dict(artifact_rep), None


def get_catalog_name_and_args(
    name: str, catalogs: Optional[List[AbstractCatalog]] = None
):
    name, args = separate_inside_and_outside_square_brackets(name)

    if catalogs is None:
        catalogs = list(Catalogs())

    for catalog in catalogs:
        if name in catalog:
            return catalog, name, args

    raise UnitxtArtifactNotFoundError(name, catalogs)


def verbosed_fetch_artifact(identifier):
    artifact, catalog = fetch_artifact(identifier)
    logger.debug(f"Artifact {identifier} is fetched from {catalog}")
    return artifact


def reset_artifacts_json_cache():
    artifacts_json_cache.cache_clear()


def maybe_recover_artifact(obj):
    if Artifact.is_possible_identifier(obj):
        return verbosed_fetch_artifact(obj)
    return obj


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
            raise UnitxtError(
                "'UNITXT_DATA_CLASSIFICATION_POLICY' should be of type "
                "'Dict[str, List[str]]', where a artifact's name is a key, and a "
                "value is a list of data classifications used by that artifact.",
                Documentation.DATA_CLASSIFICATION_POLICY,
            )

    if artifact not in data_classification.keys():
        return None

    return data_classification.get(artifact)

import json
from typing import Any, Dict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "_"
) -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def load_json(path):
    with open(path) as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError as e:
            with open(path) as f:
                file_content = "\n".join(f.readlines())
            raise RuntimeError(
                f"Failed to decode json file at '{path}' with file content:\n{file_content}"
            ) from e


def save_json(path, data):
    with open(path, "w") as f:
        dumped = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(dumped)
        f.write("\n")

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import yaml
from dacite import Config as DaciteConfig
from dacite import from_dict


@dataclass
class Config:
    @classmethod
    def from_dict[T](cls: type[T], config_dict: dict) -> T:
        config_dict = dict(config_dict)  # leave the caller's dict alone
        type_key = config_dict.pop(f"{cls.__name__.lower()}_type", None)
        config_class = cls
        if type_key is not None:
            subclasses = {sub.__name__.lower(): sub for sub in type.__subclasses__(cls)}
            if type_key not in subclasses:
                raise ValueError(
                    f"{cls.__name__} type {type_key!r} is not one of {list(subclasses)}"
                )
            config_class = subclasses[type_key]
        return from_dict(
            config_class,
            config_dict,
            config=DaciteConfig(
                strict=True,
                cast=[tuple],  # YAML has no tuples: accept a list for a tuple field
                type_hooks={
                    float: float,
                    **{config_cls: config_cls.from_dict for config_cls in all_config_classes()},
                },
            ),
        )

    def to_dict(self) -> dict:
        config_dict = {
            field.name: to_plain(getattr(self, field.name)) for field in fields(self)
        }
        parent = type(self).__bases__[0]
        if parent is not Config:
            config_dict[f"{parent.__name__.lower()}_type"] = type(self).__name__.lower()
        return config_dict


def to_plain(value):
    """Config -> its to_dict(); other dataclasses, lists and tuples -> plain dicts and lists."""
    if isinstance(value, Config):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)  # e.g. Scene.Bounds, which is not a Config
    if isinstance(value, list | tuple):
        return [to_plain(item) for item in value]
    return value


def all_config_classes(cls=Config):
    for subclass in type.__subclasses__(cls):
        yield subclass
        yield from all_config_classes(subclass)


def load_config(config_path: Path | str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config_dict: dict, save_path: Path | str) -> None:
    with open(save_path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

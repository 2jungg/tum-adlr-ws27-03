import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import yaml
from dacite import Config as DaciteConfig
from dacite import from_dict


@dataclass
class Config:
    @classmethod
    def key(cls) -> str:
        return snake_case(cls.__name__.removesuffix("Config"))

    @classmethod
    def from_dict[T: Config](cls: type[T], config_dict: dict) -> T:
        config_dict = dict(config_dict)  # leave the caller's dict alone
        type_key = config_dict.pop(f"{family_root(cls).key()}_type", None)
        config_class = cls
        if type_key is not None:
            choices = {sub.key(): sub for sub in (cls, *all_config_classes(cls))}
            if type_key not in choices:
                raise ValueError(
                    f"{cls.__name__} type {type_key!r} is not one of {list(choices)}"
                )
            config_class = choices[type_key]
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

    @classmethod
    def from_yaml[T: Config](cls: type[T], path: Path | str) -> T:
        """the file's section family_root(cls).key(), e.g. model: for MLPConfig"""
        return cls.from_dict(load_config(path)[family_root(cls).key()])

    def to_dict(self) -> dict:
        config_dict = {
            field.name: to_plain(getattr(self, field.name)) for field in fields(self)
        }
        root = family_root(type(self))
        if root is not type(self):
            config_dict[f"{root.key()}_type"] = type(self).key()
        return config_dict

    def to_yaml(self, path: Path | str) -> None:
        save_config({family_root(type(self)).key(): self.to_dict()}, path)


def snake_case(name: str) -> str:
    """CamelCase -> snake_case, acronyms kept together: MLPConfig -> mlp_config"""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def family_root(cls: type[Config]) -> type[Config]:
    """The most general Config subclass that cls derives from, e.g. Obstacle for Circle."""
    return [base for base in cls.__mro__ if issubclass(base, Config) and base is not Config][-1]


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


class ConfigDumper(yaml.SafeDumper):
    """Writes sequences of scalars on one line, e.g. center: [1.2, 0.5]."""


def _represent_sequence(dumper: ConfigDumper, sequence: list | tuple) -> yaml.Node:
    is_leaf = all(not isinstance(item, list | tuple | dict) for item in sequence)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", sequence, flow_style=is_leaf)


ConfigDumper.add_representer(list, _represent_sequence)
ConfigDumper.add_representer(tuple, _represent_sequence)


def load_config(config_path: Path | str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config_dict: dict, save_path: Path | str) -> None:
    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, Dumper=ConfigDumper, sort_keys=False)

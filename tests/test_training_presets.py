"""The built-in presets under `training_presets/`, checked as data.

A preset is a JSON file that reaches `TrainConfig` through
`TopBarController.load_config_from_file`. Two properties of that path make a
wrong preset expensive to notice:

* `BaseConfig.from_dict` iterates the *schema*, not the data. A key the schema
  does not have is neither applied nor reported -- see
  `test_a_key_train_config_does_not_have_is_silently_ignored`, which is here so
  the rest of this file's strictness has a stated reason.
* built-in presets skip migration entirely (`migrate=not is_built_in_preset`),
  so a key that only survives *because* a migration renames it is dead on
  arrival in a preset while still working in a user config.

Together that means a preset can name a field that was renamed three releases
ago, load without a murmur, and quietly train something other than what it says.
"""

import json
import os
from pathlib import Path
from typing import get_args, get_origin

from modules.ui.TopBarController import TopBarController
from modules.util import create
from modules.util.config.BaseConfig import BaseConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.type_util import issubclass_safe

import pytest

# anchored on the repo root rather than the CWD: an empty parametrize list is a
# silently passing test file, and pytest can legitimately be run from anywhere.
PRESET_DIR = str(Path(__file__).resolve().parent.parent / "training_presets")


def _preset_paths() -> list[str]:
    # "#" marks a built-in; "#.json" is the last-session state, not a preset.
    # Same convention as TopBarController.load_preset_tree.
    return sorted(
        os.path.join(dirpath, name)
        for dirpath, _, filenames in os.walk(PRESET_DIR)
        for name in filenames
        if name.startswith("#") and name.endswith(".json") and name != "#.json"
    )


PRESET_PATHS = _preset_paths()


def _ids(paths):
    return [os.path.relpath(p, PRESET_DIR) for p in paths]


def _unknown_keys(config: BaseConfig, data: dict, prefix: str = "") -> list[str]:
    """Keys in `data` that `config`'s schema has no field for, at any depth.

    Recurses into nested configs, dict-of-config fields (`optimizer_defaults`) and
    list-of-config fields (`concepts`, `samples`, `additional_embeddings`) --
    a typo is just as silent three levels down.
    """
    unknown = []
    for key, value in data.items():
        if key.startswith("__"):  # "__version", and the same reserved prefix to_dict writes
            continue
        if key not in config.types:
            unknown.append(prefix + key)
            continue

        field_type = config.types[key]
        if issubclass_safe(field_type, BaseConfig) and isinstance(value, dict):
            unknown += _unknown_keys(getattr(config, key), value, f"{prefix}{key}.")
        elif (field_type is dict or get_origin(field_type) is dict) and isinstance(value, dict):
            args = get_args(field_type)
            if len(args) > 1 and issubclass_safe(args[1], BaseConfig):
                for entry_key, entry in value.items():
                    if isinstance(entry, dict):
                        unknown += _unknown_keys(args[1].default_values(), entry, f"{prefix}{key}.{entry_key}.")
        elif (field_type is list or get_origin(field_type) is list) and isinstance(value, list):
            args = get_args(field_type)
            if args and issubclass_safe(args[0], BaseConfig):
                for index, entry in enumerate(value):
                    if isinstance(entry, dict):
                        unknown += _unknown_keys(args[0].default_values(), entry, f"{prefix}{key}[{index}].")
    return unknown


def _load(path: str) -> TrainConfig:
    """Exactly what TopBarController.load_config_from_file does for a built-in."""
    with open(path, "r") as f:
        data = json.load(f)
    return TrainConfig.default_values().from_dict(data, migrate=False).to_unpacked_config()


def test_there_are_presets_to_check():
    """A path typo here would turn every parametrized test below into zero tests."""
    assert len(PRESET_PATHS) > 20


def test_a_key_train_config_does_not_have_is_silently_ignored():
    """The premise of this whole file: a misspelled preset key raises nothing.

    `from_dict` walks `self.types`, so a key the schema lacks is never looked at.
    Nothing warns, nothing raises, and the value simply does not arrive.
    """
    config = TrainConfig.default_values()
    before = config.learning_rate

    config.from_dict({"learnign_rate": 0.5, "no_such_field_at_all": True}, migrate=False)

    assert config.learning_rate == before
    assert not hasattr(config, "no_such_field_at_all")


@pytest.mark.parametrize("path", PRESET_PATHS, ids=_ids(PRESET_PATHS))
def test_every_key_a_preset_sets_is_a_real_config_field(path):
    with open(path, "r") as f:
        data = json.load(f)

    assert _unknown_keys(TrainConfig.default_values(), data) == []


@pytest.mark.parametrize("path", PRESET_PATHS, ids=_ids(PRESET_PATHS))
def test_every_preset_names_a_trainable_model_and_a_supported_output_format(path):
    config = _load(path)

    assert config.training_method in create.supported_training_methods(config.model_type)

    supported = config.model_type.supported_output_formats(config.training_method)
    assert config.output_model_format in supported, (
        f"{config.output_model_format} is not one of {[str(f) for f in supported]}"
    )


@pytest.mark.parametrize("path", PRESET_PATHS, ids=_ids(PRESET_PATHS))
def test_every_layer_filter_preset_a_preset_names_exists_for_that_model(path):
    """A preset name is resolved against whichever model's LAYER_PRESETS is loaded.

    `layer_filter_entry` silently falls back to the first entry when the stored
    name is not in the list, so naming another family's preset does not fail --
    it trains a different set of layers than the file says.
    """
    config = _load(path)
    setup_cls = create.get_model_setup_class(config.model_type, config.training_method)
    names = list(setup_cls.LAYER_PRESETS if setup_cls is not None else {"full": []}) + ["custom"]

    assert config.layer_filter_preset in names
    assert config.quantization.layer_filter_preset in names


def test_every_model_type_a_preset_selects_is_in_the_model_type_dropdown():
    """Otherwise loading that preset skips the model-type change handler entirely.

    `options_kv` fires its `command` from a var trace that first matches the new
    value against the dropdown's own value list. A model type missing from
    `get_model_types()` matches nothing, so the command never runs -- and the
    command is what rebinds the layer-filter widgets to the new model's
    LAYER_PRESETS. The preset then loads with the *previous* model's layer filter
    still in place, which is a training run that quietly adapts the wrong modules
    (or, if no name matches at all, dies at setup with "no modules were matched").
    """
    listed = {model_type for _, model_type in TopBarController.get_model_types(None)}
    selected = {_load(path).model_type for path in PRESET_PATHS}

    assert selected <= listed, f"presets select model types the dropdown does not offer: {selected - listed}"


def test_the_preset_tree_offers_exactly_the_preset_files_that_exist():
    """The dropdown is built by walking the directory, so it must round-trip."""

    def leaves(nodes):
        found = []
        for _, target in nodes:
            found += [target] if isinstance(target, str) else leaves(target)
        return found

    offered = leaves(TopBarController(TrainConfig.default_values()).load_preset_tree(PRESET_DIR))

    assert sorted(os.path.normpath(p) for p in offered) == [os.path.normpath(p) for p in PRESET_PATHS]

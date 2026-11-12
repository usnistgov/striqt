"""runtime docstrings substituted into the griffe tree by `docs/griffe_striqt.py`"""

import inspect
from pathlib import Path
from typing import Any

import msgspec
import pytest

import striqt.analysis
import striqt.sensor
from striqt.analysis.specs.doc import describe_msgspec_fields

griffe = pytest.importorskip('griffe')

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = f'{ROOT / "docs" / "griffe_striqt.py"}:StriqtRuntimeDocs'
PLACEHOLDERS = ('{args}', '{dtype}', '{dims}', '{coords}', '{attrs}')


def runtime_path(obj: Any) -> str:
    return f'{obj.__module__}.{obj.__qualname__}'


@pytest.fixture(scope='module')
def tree() -> griffe.ModulesCollection:
    loader = griffe.GriffeLoader(
        search_paths=[ROOT / 'src'],
        extensions=griffe.load_extensions(EXTENSION),
        docstring_parser=griffe.Parser.google,
    )
    loader.load('striqt.analysis')
    loader.load('striqt.sensor')
    loader.resolve_aliases()
    return loader.modules_collection


# %% on_function_instance
def test_measurement_docstring_matches_runtime(tree: griffe.ModulesCollection) -> None:
    func = tree[runtime_path(striqt.analysis.spectrogram)]

    assert func.docstring.value == inspect.cleandoc(striqt.analysis.spectrogram.__doc__)
    assert not any(p in func.docstring.value for p in PLACEHOLDERS)


def test_measurement_parameters_follow_runtime_signature(
    tree: griffe.ModulesCollection,
) -> None:
    func = tree[runtime_path(striqt.analysis.spectrogram)]
    names = [p.name for p in func.parameters]
    fields = {f.name for f in msgspec.structs.fields(striqt.analysis.specs.Spectrogram)}

    assert names == list(inspect.signature(striqt.analysis.spectrogram).parameters)
    assert fields <= set(names)


def test_every_registered_measurement_is_filled(tree: griffe.ModulesCollection) -> None:
    for info in striqt.analysis.registry.values():
        func = tree[runtime_path(info.func)]
        assert func.docstring.value == inspect.getdoc(info.func), info.name
        assert not any(p in func.docstring.value for p in PLACEHOLDERS), info.name


def test_measurement_docstring_parses_into_sections(
    tree: griffe.ModulesCollection,
) -> None:
    # a docstring substituted without the loader's parser renders as one text block
    sections = tree[runtime_path(striqt.analysis.spectrogram)].docstring.parsed
    kinds = {s.kind.value: s for s in sections}

    assert 'parameters' in kinds
    assert 'admonition' in kinds and kinds['admonition'].title == 'Output'
    assert {p.name for p in kinds['parameters'].value} >= {
        f.name for f in msgspec.structs.fields(striqt.analysis.specs.Spectrogram)
    }


def test_unregistered_function_keeps_static_docstring(
    tree: griffe.ModulesCollection,
) -> None:
    dump = striqt.analysis.lib.io.dump
    func = tree[runtime_path(dump)]

    assert func.docstring.value == inspect.cleandoc(dump.__doc__)
    assert [p.name for p in func.parameters] == list(inspect.signature(dump).parameters)


# %% on_class_members
def test_spec_field_takes_meta_description_and_type(
    tree: griffe.ModulesCollection,
) -> None:
    descriptions, types = describe_msgspec_fields(striqt.sensor.specs.SoapyCapture)
    cls = tree['striqt.sensor.specs.structs.SoapyCapture']
    attr = cls.members['center_frequency']

    assert attr.docstring.value == descriptions['center_frequency']
    assert attr.annotation == types['center_frequency']


def test_spec_field_docstring_shows_through_alias(
    tree: griffe.ModulesCollection,
) -> None:
    descriptions, _ = describe_msgspec_fields(striqt.sensor.specs.SensorCapture)
    alias = tree['striqt.sensor.specs.SensorCapture']

    assert isinstance(alias, griffe.Alias)
    assert alias.members['port'].docstring.value == descriptions['port']


# %% on_attribute_instance
def test_binding_attribute_names_backend_and_spec_types(
    tree: griffe.ModulesCollection,
) -> None:
    binding = striqt.sensor.bindings.warmup
    attr = tree['striqt.sensor.bindings.warmup']

    assert attr.docstring is not None
    assert f'`{binding.sensor.source_cls.__name__}`' in attr.docstring.value
    assert f'`{binding.schema.capture.__qualname__}`' in attr.docstring.value
    assert 'sensor_binding: warmup' in attr.docstring.value

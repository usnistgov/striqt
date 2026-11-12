"""each console script in ``[project.scripts]`` is a click command with a description"""

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = None

PYPROJECT = Path(__file__).parent.parent / 'pyproject.toml'


def console_script_targets() -> list[str]:
    if tomllib is None:
        return []
    with open(PYPROJECT, 'rb') as stream:
        scripts = tomllib.load(stream)['project']['scripts']
    return sorted(scripts.values())


# %% cli


@pytest.mark.skipif(tomllib is None, reason='tomllib requires python >= 3.11')
def test_console_scripts_are_listed() -> None:
    targets = console_script_targets()
    assert len(targets) >= 8, targets
    assert all(t.startswith('striqt.cli.') for t in targets), targets


@pytest.mark.parametrize('target', console_script_targets())
def test_cli_help_describes_the_command(target: str) -> None:
    import importlib

    import click

    module_name, attr = target.split(':')
    cli = getattr(importlib.import_module(module_name), attr)

    assert isinstance(cli, click.Command)
    assert isinstance(cli.help, str) and cli.help.strip(), f'{target} has no help'
    # a description passed as the command name has spaces and leaves `help` empty
    assert ' ' not in (cli.name or ''), f'{target} name {cli.name!r} is a description'

"""each console script in ``[project.scripts]`` is a click command with a description"""

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = None

PYPROJECT = Path(__file__).parent.parent / 'pyproject.toml'
CLI_DIR = Path(__file__).parent.parent / 'src' / 'striqt' / 'cli'


def console_script_targets() -> list[str]:
    if tomllib is None:
        return []
    with open(PYPROJECT, 'rb') as stream:
        scripts = tomllib.load(stream)['project']['scripts']
    return sorted(scripts.values())


def cli_modules() -> set[str]:
    """return the qualified name of each module and package under `striqt.cli`"""
    names = {p.stem for p in CLI_DIR.glob('*.py') if p.stem != '__init__'}
    names |= {p.parent.name for p in CLI_DIR.glob('*/__init__.py')}
    return {f'striqt.cli.{name}' for name in names}


# %% cli


@pytest.mark.skipif(tomllib is None, reason='tomllib requires python >= 3.11')
def test_console_scripts_are_listed() -> None:
    targets = console_script_targets()
    assert {t.split(':')[0] for t in targets} == cli_modules()


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

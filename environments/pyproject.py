"""merge conda environment files to avoid manual duplication of inherited dependencies"""

#!python3

from ruamel.yaml import YAML  # use of ruamel.yaml preserves comments
import sys
import re
import toml
from pathlib import Path
from functools import lru_cache

CONDA_SKIP = {'python', 'pip', 'mscorefonts', 'soapysdr', 'soapysdr-module-airt', 'pigz'}
DEV_DEPENDENCIES = {'jupyter', 'ruff', 'toml', 'ruamel.yaml'}
PYPI_RENAME = {'ruamel.yaml': 'ruamel_yaml', 'matplotlib-base': 'matplotlib'}

ENV_DIR = Path(__file__).parent
PYPROJECT_PATH = ENV_DIR.parent / 'pyproject.toml'

yaml = YAML()


@lru_cache(100)
def read_layer(recipe_path, layer_relative_path):
    return yaml.load(open(recipe_path.parent / layer_relative_path, 'r'))


def package_name(s):
    result = re.split(r'\ *\~*[\@\=\>\<\ ]+', s, maxsplit=1)[0]
    if result in 'python-freethreading':
        return 'python'
    else:
        return result


def ordered_merge(l1, l2):
    return list(dict.fromkeys(l1 + l2).keys())


def ordered_dependency_merge(l1, l2):
    # avoid duplicating a specific package name
    d1 = {package_name(s): s for s in l1}
    d2 = {package_name(s): s for s in l2}
    return list({**d1, **d2}.values())


def pop_pip(dependency_list):
    for i, obj in enumerate(dependency_list):
        if isinstance(obj, dict):
            return dependency_list.pop(i)['pip']
    return None


def rename_pypi_deps(dep_map):
    out = dict(dep_map)
    for name, replace in PYPI_RENAME.items():
        if name in dep_map:
            out[name] = out[name].replace(name, replace)
    return out


pyproject = toml.load(PYPROJECT_PATH)
deps_base = None
# merge layers in recipes
for recipe_file in ('cpu.yml', 'gpu-cpu.yml', 'edge-airt.yml'):
    recipe_name = recipe_file.rsplit('.', 1)[0]
    env = yaml.load(open(ENV_DIR / recipe_file, 'r'))

    # list of pip dependencies, without any editable install lines
    pip_deps = pop_pip(env['dependencies']) or []
    pip_deps = [s for s in pip_deps if not s.startswith(('-','http'))]

    yml_deps = {package_name(p): p for p in env['dependencies']}
    yml_deps = {k: yml_deps[k] for k in yml_deps.keys() - CONDA_SKIP}

    # split the packages into development and base dependencies
    deps = {k: yml_deps[k] for k in yml_deps.keys() - DEV_DEPENDENCIES}
    deps = rename_pypi_deps(deps)
    deps = sorted(list(deps.values()) + pip_deps)

    if deps_base is not None:
        deps = sorted(set(deps) - set(deps_base))

    if recipe_file == 'cpu.yml':
        pyproject['project'].update(dependencies=deps)
        deps_base = deps
    else:
        pyproject['project']['optional-dependencies'][recipe_name] = deps

    if recipe_file == 'cpu.yml':
        deps_dev = {k: yml_deps[k] for k in (yml_deps.keys() & DEV_DEPENDENCIES)}
        deps_dev = rename_pypi_deps(deps_dev)
        deps_dev_list = sorted(deps_dev.values())
        pyproject['project']['optional-dependencies'].update(dev=deps_dev_list)

with open(PYPROJECT_PATH, 'w') as fd:
    encoder = toml.TomlArraySeparatorEncoder(separator=',\n')
    toml.dump(pyproject, fd, encoder=encoder)

print('wrote updated dependencies to pyproject.toml')
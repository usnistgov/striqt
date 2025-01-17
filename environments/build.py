"""merge conda environment files to avoid manual duplication of inherited dependencies"""

from ruamel.yaml import YAML  # use of ruamel.yaml preserves comments
import sys
import re
from pathlib import Path
from functools import lru_cache

RECIPE_DIR = Path(__file__).parent / 'recipes'
OUTPUT_DIR = Path(__file__).parent

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


# find recipes
recipe_paths = list(RECIPE_DIR.glob('*.yml')) + list(RECIPE_DIR.glob('*.yaml'))
if len(recipe_paths) == 0:
    print('no recipes', file=sys.stderr)

# merge layers in recipes
for recipe_path in recipe_paths:
    recipe = yaml.load(open(recipe_path, 'r'))
    env = read_layer(recipe_path, recipe['inherit'][0])
    env_pip = pop_pip(env['dependencies']) or []

    for layer_path in recipe['inherit'][1:]:
        layer = read_layer(recipe_path, layer_path)

        if 'name' in layer:
            env.setdefault('name', layer)

        env['channels'] = ordered_merge(
            layer.get('channels', []), env.get('channels', [])
        )

        new_pip = pop_pip(layer['dependencies'])

        env['dependencies'] = ordered_dependency_merge(
            env.get('dependencies', []), layer.get('dependencies', [])
        )

        if new_pip is not None:
            env_pip = ordered_dependency_merge(env_pip, new_pip)

    if len(env_pip) > 0:
        env['dependencies'].append({'pip': env_pip})

    yaml.dump(env, open(OUTPUT_DIR / recipe_path.name, 'w'))
    print(f"wrote '{OUTPUT_DIR / recipe_path.name}'")

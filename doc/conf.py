import builtins
from pathlib import Path
from typing import Union
import typing
import types
import msgspec
import re
import inspect

import numpy as np
import toml
from sphinx.domains.python import PythonDomain
from sphinx.ext import autodoc
from sphinx.util.inspect import stringify_signature, stringify_annotation

import striqt.sensor as module
import sys
import importlib.util


def reify_all_lazy_modules():
    """Iterates through sys.modules and reifies any LazyLoader proxies."""
    # We use list(sys.modules.items()) to avoid "dictionary changed size during iteration"
    # because reifying a module might trigger further imports.
    for name, module in list(sys.modules.items()):
        # Check if the module is a lazy proxy. 
        # LazyLoader modules typically have a __spec__ with a LazyLoader loader.
        spec = getattr(module, '__spec__', None)
        if spec and isinstance(spec.loader, importlib.util.LazyLoader):
            # Accessing any attribute (like __name__) triggers the actual load.
            _ = module.__name__

reify_all_lazy_modules()


# load and validate the project definition from pyproject.toml
project_info = toml.load('../pyproject.toml')
missing_fields = {'name'} - set(project_info['project'].keys())
if len(missing_fields) > 0:
    raise ValueError(
        f'fields {missing_fields} missing from [project] in pyproject.toml'
    )

# Location of the API source code
autoapi_dirs = [f'../src/{project_info["project"]["name"]}']
if not Path(autoapi_dirs[0]).exists():
    raise OSError(f'did not find source directory at expected path "{autoapi_dirs[0]}"')

# -------- General information about the project ------------------
project = project_info['project']['name']

if 'authors' in project_info['project']:
    authors = [author['name'] for author in project_info['project']['authors']]
    author_groups = [
        ', '.join(a) for a in np.array_split(authors, np.ceil(len(authors) / 3))
    ]
else:
    author_groups = []

copyright = (
    'United States government work, not subject to copyright in the United States'
)
version = release = module.__version__
language = 'en'

# ------------- base sphinx setup -------------------------------
extensions = [
    #
    # base sphinx capabilities
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    #
    # handles notebooks
    'myst_nb',
    #
    # numpy- and google-style docstrings
    'sphinx.ext.napoleon',
    #
    # for code that will be hosted on github pages (or NIST pages)
    'sphinx.ext.githubpages',
]

exclude_patterns = [
    '_build',
    'jupyter_execute',
    f'{project}/__about__.py',
    '**.ipynb_checkpoints',
    'setup*',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Force handlers
source_suffix = {
    '.rst': 'restructuredtext',
    # '.ipynb': 'myst-nb',
    # '.md': 'myst-nb',
}

autodoc_mock_imports = []

# The master toctree document.
master_doc = 'index'


# ------------------ myst_nb ---------------------------------------
# For debug: uncomment this
# nb_execution_mode = "off"

# merge consecutive notebook logger outputs into shared text box
nb_merge_streams = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'
todo_include_todos = False

# ------------- HTML output ----------------------------------------
html_theme = 'pyramid'
html_title = f'{project}'
html_static_path = ['_static']
html_use_index = False
html_show_sphinx = False
html_theme_options = {'sidebarwidth': '28em'}
htmlhelp_basename = project + 'doc'

# ------ LaTeX output ---------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'\setcounter{tocdepth}{5}',
}

latex_documents = [
    (
        master_doc,
        f'{project}-api.tex',
        rf'API reference for {project}',
        r', \\'.join(author_groups),
        'manual',
    ),
]
latex_show_urls = 'False'
latex_domain_indices = False

# ------------- misc ---------------------------------------------
mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}


# -- Dynamic processing to get the library introspection right ----
class PatchedPythonDomain(PythonDomain):
    """avoid clobbering references to builtins"""

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        # ref: https://github.com/sphinx-doc/sphinx/issues/3866#issuecomment-311181219
        exclude_targets = set(dir(builtins))

        if 'refspecific' in node:
            if not node['refspecific'] and node['reftarget'] in exclude_targets:
                del node['refspecific']

        return super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode
        )


class ClassDocumenter(autodoc.ClassDocumenter):
    def get_object_members(self, want_all: bool):
        _, members = super().get_object_members(True)
        members = self.filter_members(members, want_all)

        return members

def parse_type_and_meta(hint):
    """
    Recursively walks a type hint to extract msgspec.Meta objects.
    Safely handles Literal values, Ellipsis, Unions, and generic containers.
    """
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    
    if origin is typing.Annotated:
        clean_base, metas = parse_type_and_meta(args[0])
        msgspec_metas = [m for m in args[1:] if isinstance(m, msgspec.Meta)]
        return clean_base, msgspec_metas + metas
        
    if origin is typing.Literal:
        return hint, []
        
    if args:
        cleaned_args = []
        all_metas = []
        
        for arg in args:
            if arg is Ellipsis:
                cleaned_args.append(Ellipsis)
            else:
                clean_arg, metas = parse_type_and_meta(arg)
                cleaned_args.append(clean_arg)
                all_metas.extend(metas)
        
        try:
            if hasattr(types, "UnionType") and origin is types.UnionType:
                import operator
                from functools import reduce
                clean_type = reduce(operator.or_, cleaned_args)
            else:
                clean_type = origin[tuple(cleaned_args)]
        except Exception:
            clean_type = hint
            
        return clean_type, all_metas
        
    return hint, []


def clean_type_string(type_str):
    """Applies string replacements to clean up Literal and module paths."""
    # 1. Clean up Literal
    type_str = re.sub(r"\bLiteral\[", "one of [", type_str)
    
    # # 2. Remap nested module paths to top-level paths
    # for old_path, new_path in MODULE_REMAPS.items():
    #     type_str = type_str.replace(old_path, new_path)
        
    return type_str


def extract_msgspec_meta(app, what, name, obj, options, lines):
    """
    Injects dynamic parameters for struct fields and a separate list 
    for ClassVar constants into the class docstring, extracting Meta descriptions for both.
    """
    if what == "class" and isinstance(obj, type) and issubclass(obj, msgspec.Struct):
        try:
            hints = typing.get_type_hints(obj, include_extras=True)
            struct_fields = {f.name for f in msgspec.structs.fields(obj)}
        except Exception:
            return

        if lines and lines[-1] != "":
            lines.append("")

        for field_name, field_type in hints.items():
            
            # --- 1. HANDLE MSGSPEC STRUCT FIELDS ---
            if field_name in struct_fields:
                clean_type, metas = parse_type_and_meta(field_type)
                
                description = ""
                constraints = []
                
                for meta in metas:
                    if meta.description and not description:
                        description = meta.description
                    
                    if getattr(meta, 'gt', None) is not None: constraints.append(f"> {meta.gt}")
                    if getattr(meta, 'ge', None) is not None: constraints.append(f">= {meta.ge}")
                    if getattr(meta, 'lt', None) is not None: constraints.append(f"< {meta.lt}")
                    if getattr(meta, 'le', None) is not None: constraints.append(f"<= {meta.le}")
                    if getattr(meta, 'multiple_of', None) is not None: constraints.append(f"multiple of {meta.multiple_of}")
                    if getattr(meta, 'min_length', None) is not None: constraints.append(f"min_length={meta.min_length}")
                    if getattr(meta, 'max_length', None) is not None: constraints.append(f"max_length={meta.max_length}")
                    if getattr(meta, 'pattern', None) is not None: constraints.append(f"pattern='{meta.pattern}'")
                    
                    if meta.extra and 'units' in meta.extra:
                        constraints.append(f"units: {meta.extra['units']}")

                if constraints:
                    constraint_str = f" *(Constraints: {', '.join(constraints)})*"
                    if description:
                        description += constraint_str
                    else:
                        description = constraint_str.strip()
                
                raw_type_str = stringify_annotation(clean_type)
                final_type_str = clean_type_string(raw_type_str)

                lines.append(f":param {field_name}: {description}")
                lines.append(f":type {field_name}: {final_type_str}")

            # --- 2. HANDLE CLASSVAR CONSTANTS ---
            else:
                origin = typing.get_origin(field_type)
                if field_type is typing.ClassVar or origin is typing.ClassVar:
                    # Extract the underlying type if it's parameterized
                    if origin is typing.ClassVar:
                        args = typing.get_args(field_type)
                        inner_type = args[0] if args else typing.Any
                    else:
                        inner_type = typing.Any
                    
                    # NEW: Run it through our parser to strip Annotated and grab Meta
                    clean_type, metas = parse_type_and_meta(inner_type)
                    
                    # Extract description from Meta
                    description = ""
                    for meta in metas:
                        if meta.description and not description:
                            description = meta.description
                    
                    # Fetch the actual constant value from the class object
                    val = getattr(obj, field_name, None)
                    val_str = f"*(Value: {val})*" if val is not None else ""
                    
                    # Combine description and value cleanly
                    if description and val_str:
                        final_desc = f"{description} {val_str}"
                    elif description:
                        final_desc = description
                    else:
                        final_desc = val_str
                    
                    # Stringify the stripped, clean type
                    raw_type_str = stringify_annotation(clean_type)
                    final_type_str = clean_type_string(raw_type_str)
                    
                    lines.append(f":cvar {field_name}: {final_desc}")
                    lines.append(f":vartype {field_name}: {final_type_str}")
                    
def simplify_msgspec_signature(app, what, name, obj, options, signature, return_annotation):
    """
    Strips type annotations from the class __init__ signature AND 
    evaluates default_factory functions to show the actual default values.
    """
    if what == "class" and isinstance(obj, type) and issubclass(obj, msgspec.Struct):
        try:
            sig = inspect.signature(obj)
            
            # Map field names to their FieldInfo objects to access default_factory
            struct_fields = {f.name: f for f in msgspec.structs.fields(obj)}
            
            new_parameters = []
            for param_name, param in sig.parameters.items():
                # 1. Strip the type annotation to keep the signature clean
                clean_param = param.replace(annotation=inspect.Parameter.empty)
                
                # 2. Check for and execute default_factory
                field_info = struct_fields.get(param_name)
                if field_info and field_info.default_factory is not msgspec.NODEFAULT:
                    try:
                        # Run the factory function to get the generated default value
                        generated_default = field_info.default_factory()
                        
                        # Replace the default in the signature with the generated value
                        clean_param = clean_param.replace(default=generated_default)
                    except Exception:
                        # If the factory requires arguments or fails dynamically, 
                        # safely fallback to Sphinx's default rendering
                        pass
                        
                new_parameters.append(clean_param)
            
            new_sig = sig.replace(parameters=new_parameters)
            return stringify_signature(new_sig), return_annotation
        except (ValueError, TypeError):
            pass

def skip_msgspec_struct_fields(app, what, name, obj, skip, options):
    """
    Forces Sphinx to skip documenting msgspec fields as standalone attributes,
    and universally skips any attributes type-hinted as typing.ClassVar.
    """
    if what == "class":
        mod_name = app.env.temp_data.get('autodoc:module')
        cls_name = app.env.temp_data.get('autodoc:class')

        if mod_name and cls_name:
            try:
                mod = sys.modules.get(mod_name)
                if not mod:
                    return skip

                cls = mod
                for part in cls_name.split('.'):
                    cls = getattr(cls, part)

                if isinstance(cls, type):
                    # 1. NEW: Universally skip ClassVar attributes
                    try:
                        hints = typing.get_type_hints(cls, include_extras=True)
                        if name in hints:
                            hint = hints[name]
                            # Handle both parameterized (ClassVar[int]) and bare (ClassVar)
                            if hint is typing.ClassVar or typing.get_origin(hint) is typing.ClassVar:
                                return True
                    except Exception:
                        pass

                    # 2. EXISTING: Skip msgspec.Struct instance fields
                    if issubclass(cls, msgspec.Struct):
                        struct_fields = {f.name for f in msgspec.structs.fields(cls)}

                        if name in struct_fields:
                            return True

                if "rx_enable_delay" in name:
                    print(f"\n--- SPHINX DIAGNOSTIC START ---")
                    print(mod_name, cls_name)
                    print(f"Target Name : {name}")
                    print(f"Target What : {what}")
                    print(f"Mod : {mod!r}")
                    print(f"Target Obj  : {type(obj)}")
                    print(f'What: {what!r}')
                    
                    cls = app.env.temp_data.get('autodoc:class')
                    print(f"Temp Class  : {cls!r}")
                    # print(f"cls fields: ", repr(cls))
                    if cls:
                        is_struct = isinstance(cls, type) and issubclass(cls, msgspec.Struct)
                        print(f"Is Struct?  : {is_struct}")
                    else:
                        print(f"Is Struct?  : N/A (cls is None)")

                    print(f"Current Skip: {skip}")
                    print(f"--- SPHINX DIAGNOSTIC END ---\n")

            except Exception:
                raise

    return skip

def setup(app):
    app.add_domain(PatchedPythonDomain, override=True)
    app.add_autodocumenter(ClassDocumenter, override=True)
    app.connect('autodoc-process-docstring', extract_msgspec_meta)
    app.connect('autodoc-process-signature', simplify_msgspec_signature)
    app.connect('autodoc-skip-member', skip_msgspec_struct_fields)
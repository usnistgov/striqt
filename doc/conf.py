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

import striqt.sensor as ss
from striqt.sensor.lib.controller import Controller
from striqt.analysis.specs import doc as spec_doc
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
version = release = ss.__version__
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

autodoc_default_options = {
    'imported-members': True,
    'members': True,
    'undoc-members': True,  # Ensures instances without docstrings are still evaluated
}

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

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode): # type: ignore
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
        # Capture BOTH the success boolean and the members list
        success, members = super().get_object_members(True)

        # Filter the members
        members = self.filter_members(members, want_all)

        # Return the expected tuple back to Sphinx
        return success, members


def clean_type_string(type_str):
    """Applies string replacements to clean up Literal and module paths."""
    # 1. Clean up Literal
    type_str = re.sub(r'\bLiteral\[', 'one of [', type_str)

    # # 2. Remap nested module paths to top-level paths
    # for old_path, new_path in MODULE_REMAPS.items():
    #     type_str = type_str.replace(old_path, new_path)

    return type_str


def extract_msgspec_meta(app, what, name, obj, options, lines):
    """
    Injects dynamic parameters for struct fields and a separate list
    for ClassVar constants into the class docstring, extracting Meta descriptions for both.
    """
    if what == 'class' and isinstance(obj, type) and issubclass(obj, msgspec.Struct):
        try:
            hints = typing.get_type_hints(obj, include_extras=True)
            struct_fields = {f.name for f in msgspec.structs.fields(obj)}
        except Exception:
            return

        if lines and lines[-1] != '':
            lines.append('')

        field_descriptions, field_types = spec_doc.describe_msgspec_fields(obj)

        for field_name, field_type in hints.items():
            if field_name in struct_fields:
                desc = field_descriptions.get(field_name, "")
                type_str = field_types.get(field_name, "Any")

                lines.append(f":param {field_name}: {desc}")
                lines.append(f":type {field_name}: {type_str}")

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
                    clean_type, metas = spec_doc.parse_type_and_meta(inner_type)

                    # Extract description from Meta
                    description = ''
                    for meta in metas:
                        if meta.description and not description:
                            description = meta.description

                    # Fetch the actual constant value from the class object
                    val = getattr(obj, field_name, None)
                    val_str = f'*(Value: {val})*' if val is not None else ''

                    # Combine description and value cleanly
                    if description and val_str:
                        final_desc = f'{description} {val_str}'
                    elif description:
                        final_desc = description
                    else:
                        final_desc = val_str

                    # Stringify the stripped, clean type
                    raw_type_str = stringify_annotation(clean_type)
                    final_type_str = clean_type_string(raw_type_str)

                    lines.append(f':cvar {field_name}: {final_desc}')
                    lines.append(f':vartype {field_name}: {final_type_str}')


def simplify_msgspec_signature(
    app, what, name, obj, options, signature, return_annotation
):
    """
    Strips type annotations from the class __init__ signature AND
    evaluates default_factory functions to show the actual default values.
    """
    if what == 'class' and isinstance(obj, type) and issubclass(obj, msgspec.Struct):
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
    if what == 'class':
        mod_name = app.env.temp_data.get('autodoc:module')
        cls_name = app.env.temp_data.get('autodoc:class')

        if mod_name and cls_name:
            mod = sys.modules.get(mod_name)
            if not mod:
                return skip

            cls = mod
            for part in cls_name.split('.'):
                cls = getattr(cls, part)

            if isinstance(cls, type):
                try:
                    hints = typing.get_type_hints(cls, include_extras=True)
                    if name in hints:
                        hint = hints[name]
                        # Handle both parameterized (ClassVar[int]) and bare (ClassVar)
                        if (
                            hint is typing.ClassVar
                            or typing.get_origin(hint) is typing.ClassVar
                        ):
                            return True
                except Exception:
                    pass

                if issubclass(cls, msgspec.Struct):
                    struct_fields = {f.name for f in msgspec.structs.fields(cls)}

                    if name in struct_fields:
                        return True


def skip_external_imports(app, what, name, obj, skip, options):
    """
    Forces Sphinx to skip objects imported from outside the current module hierarchy.
    Bypasses Sphinx sentinels to evaluate the true underlying Python object.
    """
    # 1. Do not interfere with class members
    if app.env.temp_data.get('autodoc:class'):
        return None

    current_module_name = app.env.temp_data.get('autodoc:module')
    if not current_module_name:
        return None

    # 2. Fetch the live module to bypass Sphinx sentinels
    import sys

    mod = sys.modules.get(current_module_name)
    if not mod:
        return None

    short_name = name.split('.')[-1]

    # 3. Extract the REAL object directly from the module
    real_obj = getattr(mod, short_name, obj)

    if real_obj is getattr(typing, short_name, None):
        # skip typing objects that sometimes have their
        # __module__ renamed
        return True

    # 4. Check the real object's native module
    obj_module_name = getattr(real_obj, '__module__', None)
    type_module_name = getattr(type(real_obj), '__module__', None)

    # # If it genuinely originated from outside our current module tree, nuke it
    # if obj_module_name and not obj_module_name.startswith(current_module_name):
    #     return True

    # Typing module objects are weird; catch them by their origin or their type
    if obj_module_name == 'typing' or type_module_name == 'typing':
        return True

    # Absolute last resort for uncooperative typing singletons
    if short_name in {'Union', 'Optional', 'Any', 'Literal', 'Annotated', 'ClassVar'}:
        return True

    return None


def list_sensor_bindings_in_module(app, what, name, obj, options, lines):
    """
    Dynamically injects a list of SensorBinding instances into the module's docstring,
    recursively unpacking their dataclass/struct fields and cross-referencing types.
    """
    if what == 'module':
        try:
            mod = sys.modules.get(name)
            if not mod:
                return

            bindings = {}
            for attr_name, attr_val in vars(mod).items():
                if isinstance(
                    attr_val, ss.lib.bindings.SensorBinding
                ) and not isinstance(attr_val, type):
                    bindings[attr_name] = attr_val

            if bindings:
                if lines and lines[-1] != '':
                    lines.append('')

                lines.append('**Available Sensor Bindings:**')
                lines.append('')

                def _unpack_fields(target_obj, indent='  '):
                    try:
                        if hasattr(target_obj, '__dataclass_fields__'):
                            fields = {
                                f: getattr(target_obj, f)
                                for f in target_obj.__dataclass_fields__
                            }
                        elif hasattr(msgspec, 'Struct') and isinstance(
                            target_obj, msgspec.Struct
                        ):
                            fields = {
                                f.name: getattr(target_obj, f.name)
                                for f in msgspec.structs.fields(target_obj)
                            }
                        else:
                            fields = vars(target_obj)
                    except Exception:
                        return

                    for f_name, f_val in fields.items():
                        if f_name.startswith('_'):
                            continue

                        # 2. If it's a class reference, create a Sphinx cross-reference link!
                        if isinstance(f_val, type):
                            # Build the full module path so Sphinx knows exactly where to look
                            full_path = f'{f_val.__module__}.{f_val.__name__}'

                            # The tilde (~) tells Sphinx to link the full path but only render the short name
                            lines.append(
                                f'{indent}* **{f_name}**: :class:`~{full_path}`'
                            )

                        # 3. If it's a nested dataclass or struct, RECURSE!
                        elif hasattr(f_val, '__dataclass_fields__') or (
                            hasattr(msgspec, 'Struct')
                            and isinstance(f_val, msgspec.Struct)
                        ):
                            lines.append(f'{indent}* **{f_name}**:')
                            lines.append('')
                            _unpack_fields(f_val, indent + '  ')

                        # 4. Otherwise, print the standard value
                        else:
                            lines.append(f'{indent}* **{f_name}**: ``{repr(f_val)}``')

                    lines.append('')

                for b_name, b_val in sorted(bindings.items()):
                    lines.append(f'* **{b_name}**')
                    lines.append('')
                    _unpack_fields(b_val, indent='  ')

        except Exception:
            pass


def skip_inherited_methods(app, what, name, obj, skip, options):
    """
    Forces Sphinx to skip methods on specific subclasses if those methods
    are simply inherited from the parent and not overridden.
    """
    from striqt.sensor.lib.typing import SourceBackend
    from striqt.sensor.specs import structs

    if what == 'class':
        if (
            name.startswith('_')
            or not hasattr(obj, '__qualname__')
            or not hasattr(obj, '__module__')
        ):
            return
        full_name = obj.__module__ + '.' + obj.__qualname__.rsplit('.')[0]
        parts = full_name.split('.')
        cls = sys.modules.get(parts[0])
        for part in parts[1:]:
            cls = getattr(cls, part)

        if not isinstance(cls, type):
            pass
        elif not issubclass(cls, SourceBackend):
            pass
        elif cls is SourceBackend:
            pass
        elif hasattr(SourceBackend, name):
            return True

    return None


def document_dynamic_controllers(app, what, name, obj, options, lines):
    """
    Finds dynamic subclasses of Controller in a module and injects them as fully 
    documented classes, conventionally appending the __init__ signature to the class header.
    """
    if what == "module":
        try:
            mod = sys.modules.get(name)
            if not mod:
                return
            
            dynamic_controllers = {}
            for attr_name, attr_val in vars(mod).items():
                if (isinstance(attr_val, type) and 
                    issubclass(attr_val, Controller) and 
                    attr_val is not Controller):
                    
                    dynamic_controllers[attr_name] = attr_val
                    
            if not dynamic_controllers:
                return
                
            if lines and lines[-1] != "":
                lines.append("")
                
            for c_name, c_cls in sorted(dynamic_controllers.items()):
                lines.append("---")
                lines.append("")
                
                # --- 1. Extract the constructor signature ---
                try:
                    # Inspect the class object to get the dynamic __init__ signature
                    sig = inspect.signature(c_cls)
                    clean_params = []
                    for p_name, param in sig.parameters.items():
                        if p_name == 'self':
                            continue
                        clean_params.append(param.replace(annotation=inspect.Parameter.empty))
                        
                    sig = sig.replace(parameters=clean_params, return_annotation=inspect.Signature.empty)
                    init_sig_str = stringify_signature(sig)
                except Exception:
                    init_sig_str = "(...)"
                
                # --- 2. Inject the Class directive WITH the signature ---
                lines.append(f".. py:class:: {c_name}{init_sig_str}")
                lines.append("")
                
                base_indent = "   "
                
                # --- 3. Add Bases and combined docstrings ---
                lines.append(f"{base_indent}**Bases:** :class:`~striqt.sensor.lib.controller.Controller`")
                lines.append("")
                
                # Fetch both the class docstring and the __init__ docstring
                c_doc = inspect.getdoc(c_cls)
                init_doc = inspect.getdoc(getattr(c_cls, '__init__', None))
                
                # Intelligently combine them (if __init__ has a unique docstring)
                combined_docs = []
                if c_doc:
                    combined_docs.append(c_doc)
                if init_doc and "Initialize self" not in init_doc and init_doc != c_doc:
                    combined_docs.append(init_doc)
                    
                for doc in combined_docs:
                    for doc_line in doc.split('\n'):
                        lines.append(f"{base_indent}{doc_line}")
                    lines.append("")
                    
                # --- 4. Extract and document ONLY the remaining methods ---
                target_methods = ['arm', 'acquire'] # __init__ is handled!
                
                for method_name in target_methods:
                    method_obj = getattr(c_cls, method_name, None)
                    if not method_obj:
                        continue
                        
                    try:
                        sig = inspect.signature(method_obj)
                        clean_params = []
                        for p_name, param in sig.parameters.items():
                            if p_name == 'self':
                                continue
                            clean_params.append(param.replace(annotation=inspect.Parameter.empty))
                            
                        sig = sig.replace(parameters=clean_params, return_annotation=inspect.Signature.empty)
                        sig_str = stringify_signature(sig)
                    except Exception:
                        sig_str = "(...)"
                        
                    # Inject the method directive
                    lines.append(f"{base_indent}.. py:method:: {method_name}{sig_str}")
                    lines.append("")
                    
                    m_doc = inspect.getdoc(method_obj)
                    if m_doc and "Initialize self" not in m_doc:
                        for doc_line in m_doc.split('\n'):
                            lines.append(f"{base_indent}   {doc_line}")
                    else:
                        lines.append(f"{base_indent}   *(No documentation provided)*")
                        
                    lines.append("")
                                        
        except Exception:
            pass

def skip_dynamic_controller_aliases(app, what, name, obj, skip, options):
    """
    Forces Sphinx to skip the default data alias output for dynamic controllers,
    since we are handling them natively in the module docstring hook.
    """
    if isinstance(obj, type) and issubclass(obj, Controller) and obj is not Controller:
        return True
    return None


def setup(app):
    app.add_domain(PatchedPythonDomain, override=True)
    app.add_autodocumenter(ClassDocumenter, override=True)
    # app.connect('autodoc-process-docstring', inject_bindings_as_classes)   
    app.connect('autodoc-process-docstring', document_dynamic_controllers) 
    app.connect('autodoc-process-docstring', list_sensor_bindings_in_module)
    app.connect('autodoc-process-docstring', extract_msgspec_meta)
    app.connect('autodoc-process-signature', simplify_msgspec_signature)
    app.connect('autodoc-skip-member', skip_inherited_methods)
    app.connect('autodoc-skip-member', skip_msgspec_struct_fields)
    app.connect('autodoc-skip-member', skip_external_imports)
    app.connect('autodoc-skip-member', skip_dynamic_controller_aliases)
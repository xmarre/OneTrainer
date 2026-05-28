import importlib
import pkgutil

__registry = {}
__optional_import_errors = {}

def get(base_cls, *args, **kwargs):
    entries = __registry.get(base_cls)
    if entries is None:
        return None
    for entry in entries:
        if entry[0] == args and entry[1] == kwargs:
            return entry[2]
    return None

def register(base_cls, cls, *args, **kwargs):
    if get(base_cls, *args, **kwargs) is not None:
        raise RuntimeError(f"{cls} already registered as an implementation of {base_cls} with the same criteria {args} {kwargs}")

    if base_cls not in __registry:
        __registry[base_cls] = []
    __registry[base_cls].append((args, kwargs, cls))

def _is_external_optional_import_error(exc: ImportError, importing_module: str, parent: str) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        missing_name = getattr(exc, "name", None)
        if missing_name is None:
            return True

        # A missing module inside OneTrainer itself is a real code error. A
        # missing optional dependency/submodule such as diffusers.pipelines.*,
        # transformers.*, optimum, etc. should not prevent the UI from opening.
        if missing_name == importing_module or missing_name.startswith(parent + ".") or missing_name.startswith("modules."):
            return False

        return True

    message = str(exc)
    # Keep local OneTrainer import errors strict. Optional model families often
    # fail here because a third-party package exists but no longer/recently does
    # not export a specific symbol.
    if "from 'modules." in message or 'from "modules.' in message:
        return False

    return True

def get_optional_import_errors() -> dict:
    return dict(__optional_import_errors)

def optional_import_error_summary() -> str:
    if not __optional_import_errors:
        return ""

    lines = []
    for module_name, exc_text in sorted(__optional_import_errors.items()):
        lines.append(f"- {module_name}: {exc_text}")
    return "\n".join(lines)

def import_dir(path: str, parent: str, optional: bool = False):
    for _finder, name, _ispkg in pkgutil.walk_packages([path], parent + "."):
        try:
            importlib.import_module(name)
        except (ImportError, ModuleNotFoundError) as e:
            if not optional or not _is_external_optional_import_error(e, name, parent):
                raise

            exc_text = f"{type(e).__name__}: {e}"
            __optional_import_errors[name] = exc_text
            print(f"Skipping optional OneTrainer module {name}: {exc_text}")

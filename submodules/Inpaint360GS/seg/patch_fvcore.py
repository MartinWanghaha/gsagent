import fvcore
import os
import sys

def apply_fvcore_patch():
    """
    Patches fvcore.common.registry to allow re-registration of objects.
    This prevents AssertionError when modules are imported multiple times 
    due to PYTHONPATH overlaps.
    """
    try:
        # Locate the registry.py file within the installed fvcore package
        target_path = os.path.join(os.path.dirname(fvcore.__file__), "common", "registry.py")
        print(f"Locating fvcore registry at: {target_path}")

        patched_code = """from typing import Any, Dict, Optional, Callable

class Registry:
    \"\"\"
    A modified Registry class that allows overwriting existing keys 
    to prevent 'already registered' errors during redundant imports.
    \"\"\"
    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        # Modified: Removed 'assert name not in self._obj_map' to allow re-registration
        self._obj_map[name] = obj

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def register(self, obj: Optional[Callable] = None, name: Optional[str] = None):
        if callable(obj):
            key = name or obj.__name__
            self._do_register(key, obj)
            return obj
        def wrapper(fn):
            key = name or fn.__name__
            self._do_register(key, fn)
            return fn
        return wrapper

    def get(self, name: str) -> Any:
        if name not in self._obj_map:
            raise KeyError(f"Object '{name}' not registered in registry '{self._name}'")
        return self._obj_map[name]

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, keys={list(self._obj_map.keys())})"

# Maintain backward compatibility with Detectron2
META_ARCH_REGISTRY = Registry("META_ARCH")
"""
        # Overwrite the original file with the patched version
        with open(target_path, "w") as f:
            f.write(patched_code)
        
        print(" Success: fvcore registry has been patched.")

    except Exception as e:
        print(f" Error applying patch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    apply_fvcore_patch()
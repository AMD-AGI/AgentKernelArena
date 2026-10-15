"""Isolate task-local Python namespaces in CPU tests that share one interpreter."""
from contextlib import contextmanager
from pathlib import Path
import sys

@contextmanager
def isolated_task_imports(task_root):
    task_root=Path(task_root).resolve()
    def owned(name,module):
        if name == 'scripts' or name.startswith('scripts.'):
            return True
        file=getattr(module,'__file__',None)
        return isinstance(file,str) and Path(file).resolve().is_relative_to(task_root)
    saved={name:module for name,module in list(sys.modules.items()) if owned(name,module)}
    for name in saved:sys.modules.pop(name,None)
    try:
        yield
    finally:
        for name,module in list(sys.modules.items()):
            if owned(name,module):sys.modules.pop(name,None)
        sys.modules.update(saved)

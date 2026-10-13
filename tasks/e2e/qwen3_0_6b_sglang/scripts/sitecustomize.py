"""Protected compute installation; no framework import until the server needs it."""
import os
if os.environ.get('AKA_INSTALL_KERNEL') == '1':
    import json
    from pathlib import Path
    from _aka_serving_bundle import install
    task = Path(__file__).resolve().parents[1]
    install(task, json.loads((task/'runtime.lock.json').read_text()))

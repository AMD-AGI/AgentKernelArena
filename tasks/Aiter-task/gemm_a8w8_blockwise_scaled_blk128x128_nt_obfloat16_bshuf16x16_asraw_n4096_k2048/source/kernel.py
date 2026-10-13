"""Initial production baseline. Replace run with your triton implementation."""
from pathlib import Path
from scripts.task_api import load_solution

_initial = load_solution(Path(__file__).parent / 'implementation', 'main.py::run')

def run(**kwargs):
    return _initial(**kwargs)

import ast
import json
from pathlib import Path

FLASK_SRC = Path("../reference_repos_flask/src/flask")

# We'll only look at a few well-understood files to start, rather than
# the entire codebase -- easier to sanity-check the results by eye.
TARGET_FILES = ["app.py", "helpers.py", "blueprints.py"]


def extract_functions_from_file(filepath: Path):
    """
    Parse a Python file and pull out every top-level and class-method
    function definition, along with its docstring (if it has one).

    We use Python's own `ast` (Abstract Syntax Tree) module -- this parses
    code the same way Python itself does internally, so we get accurate
    function boundaries and docstrings instead of fragile regex/string
    matching over source text.
    """
    source = filepath.read_text()
    tree = ast.parse(source)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:  # skip functions with no documentation -- nothing to learn from
                source_lines = ast.get_source_segment(source, node)
                functions.append({
                    "file": filepath.name,
                    "function_name": node.name,
                    "docstring": docstring,
                    "source_code": source_lines,
                })
    return functions


all_functions = []
for filename in TARGET_FILES:
    filepath = FLASK_SRC / filename
    extracted = extract_functions_from_file(filepath)
    print(f"{filename}: found {len(extracted)} documented functions")
    all_functions.extend(extracted)

output_path = Path("data/extracted_functions.json")
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "w") as f:
    json.dump(all_functions, f, indent=2)

print(f"\nTotal: {len(all_functions)} documented functions saved to {output_path}")
"""Convert the Databricks source notebooks in this folder to .ipynb pinned to serverless environment 5 and import them.

Usage: python publish.py [notebook_stem ...]   (default: all NN_*.py notebooks)
       python publish.py --package             (the product_retrieval package: source/tests as files, notebooks as notebooks)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

PROFILE = os.environ.get("RETRIEVAL_PROFILE", "DEFAULT")  # Databricks CLI profile of the target workspace
WORKSPACE_DIR = os.environ.get("RETRIEVAL_WORKSPACE_DIR") or "/Users/{}/ai-search-product-retrieval".format(
    json.loads(subprocess.run(["databricks", "-p", PROFILE, "current-user", "me", "-o", "json"], capture_output=True, check=True, text=True).stdout)["userName"]
)
ROOT = Path(__file__).parent
BUILD = ROOT / ".build"
SEP = "# COMMAND ----------"


def to_cell(chunk: str) -> dict:
    lines = chunk.strip("\n").splitlines()
    if lines and lines[0].startswith("# MAGIC"):
        body = [ln.removeprefix("# MAGIC").removeprefix(" ") for ln in lines]
        if body[0].startswith("%md"):
            text = "\n".join(body[1:] if body[0].strip() == "%md" else [body[0][3:].strip(), *body[1:]])
            return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
        return code_cell("\n".join(body))
    return code_cell("\n".join(lines))


def code_cell(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def build(src: Path) -> Path:
    text = src.read_text().removeprefix("# Databricks notebook source\n")
    cells = [to_cell(c) for c in text.split(SEP) if c.strip()]
    nb = {
        "cells": cells,
        "metadata": {
            "application/vnd.databricks.v1+notebook": {
                "environmentMetadata": {"base_environment": "", "environment_version": "5"},
                "language": "python",
                "notebookName": src.stem,
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = BUILD / f"{src.stem}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


def publish_package() -> None:
    """Upload the product_retrieval package (source, tests, pyproject) as workspace files and its notebooks as env-5 notebooks."""
    pkg = ROOT / "product_retrieval"
    target = f"{WORKSPACE_DIR}/product_retrieval"
    for path in sorted(pkg.rglob("*")):
        rel = path.relative_to(pkg)
        if path.is_dir() or "__pycache__" in rel.parts or ".pytest_cache" in rel.parts or rel.parts[0] in ("dist", "notebooks"):
            continue
        subprocess.run(["databricks", "-p", PROFILE, "workspace", "mkdirs", f"{target}/{rel.parent}".rstrip("/.")], check=True)
        subprocess.run(["databricks", "-p", PROFILE, "workspace", "import", f"{target}/{rel}", "--file", str(path),
                        "--format", "RAW", "--overwrite"], check=True, capture_output=True)
    BUILD.mkdir(exist_ok=True)
    subprocess.run(["databricks", "-p", PROFILE, "workspace", "mkdirs", f"{target}/notebooks"], check=True)
    for nb_src in sorted((pkg / "notebooks").glob("*.py")):
        nb = build(nb_src)
        subprocess.run(["databricks", "-p", PROFILE, "workspace", "import", f"{target}/notebooks/{nb_src.stem}", "--file", str(nb),
                        "--format", "JUPYTER", "--overwrite"], check=True)
    print(f"published package to {target}")


def main(stems: list[str]) -> None:
    BUILD.mkdir(exist_ok=True)
    sources = [ROOT / f"{s}.py" for s in stems] if stems else sorted(ROOT.glob("[0-9][0-9]_*.py"))
    subprocess.run(["databricks", "-p", PROFILE, "workspace", "mkdirs", WORKSPACE_DIR], check=True)
    for src in sources:
        nb = build(src)
        subprocess.run(
            ["databricks", "-p", PROFILE, "workspace", "import", f"{WORKSPACE_DIR}/{src.stem}",
             "--file", str(nb), "--format", "JUPYTER", "--overwrite"],
            check=True,
        )
        print(f"published {src.stem}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--package"]:
        publish_package()
    else:
        main(sys.argv[1:])

# Databricks notebook source
# MAGIC %md
# MAGIC # Build and publish the `product-retrieval` wheel
# MAGIC Runs the unit tests, builds the wheel from this folder's source, and copies it to a Unity Catalog volume so notebooks,
# MAGIC jobs, Model Serving and Apps can `pip install` it.

# COMMAND ----------

# MAGIC %uv pip install build pytest databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import glob
import os
import shutil
import subprocess
import sys

dbutils.widgets.text("volume_dir", "/Volumes/retail_consumer_goods/product_search/raw/wheels")
volume_dir = dbutils.widgets.get("volume_dir")
source = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
work = "/tmp/product_retrieval_build"
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(source, work, ignore=shutil.ignore_patterns("notebooks", "dist", "__pycache__", ".pytest_cache"))

# COMMAND ----------

tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=work, capture_output=True, text=True)
print(tests.stdout[-2000:], tests.stderr[-2000:])
assert tests.returncode == 0, "unit tests failed"

# COMMAND ----------

subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", f"{work}/dist", work], check=True, capture_output=True)
wheel = glob.glob(f"{work}/dist/*.whl")[0]
os.makedirs(volume_dir, exist_ok=True)
target = shutil.copy(wheel, volume_dir)
print(target)
dbutils.notebook.exit(target)

# -*- coding: utf-8 -*-
"""
week7_minimal_test_notebook.ipynb (flattened)

What this does:
- Auth to GCP
- Safe Cloud SQL connection (classicmodels) via connector + SQLAlchemy
- Schema helpers + QueryRunner tool
- Smoke tests and test-set validator
- Base Llama-3-8B load (pre-QLoRA placeholder)
"""

# Authenticate to Google Cloud
from google.colab import auth
auth.authenticate_user()

"""Project context (swap to env var in production)."""

# Set GCP project and validate access (Colab shell)
project_id = "modified-enigma-476414-h9"  # replace with env var in production
import os
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

# !gcloud config set project {project_id}
# !gcloud projects describe {project_id}

"""Installs (pin these in a requirements cell/file for real runs)."""

import sys
!{sys.executable} -m pip install --upgrade pip
!{sys.executable} -m pip install "cloud-sql-python-connector[pymysql]" SQLAlchemy==2.0.7 pymysql cryptography==41.0.0 --force-reinstall --no-cache-dir
!{sys.executable} -m pip install accelerate
!{sys.executable} -m pip install bitsandbytes
!{sys.executable} -m pip install peft
!{sys.executable} -m pip install transformers
!{sys.executable} -m pip install datasets
!{sys.executable} -m pip install trl

# Standard imports and logger setup
import os
import logging
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy import text
import pymysql
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nl2sql_db")

"""Connection params (env first, prompt fallback during dev)."""

# Set these environment variables in Colab using:
#   %env DB_USER=... %env DB_NAME=... etc (or use secrets manager)
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME", "classicmodels")

# Fallback interactive prompt if variables missing
if not INSTANCE_CONNECTION_NAME:
    from getpass import getpass
    INSTANCE_CONNECTION_NAME = input("Enter INSTANCE_CONNECTION_NAME: ").strip()
if not DB_USER:
    DB_USER = input("Enter DB_USER: ").strip()
if not DB_PASS:
    DB_PASS = getpass("Enter DB_PASS: ")

"""Connector + engine setup."""

# Create and manage Cloud SQL Connector with retry and safe teardown
from google.api_core import retry
from sqlalchemy.engine import Engine
import time

connector = Connector()

def getconn():
    """
    Creator function returned to SQLAlchemy. This function will be called by SQLAlchemy
    to create new DB connections using the Cloud SQL Connector.
    """
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pymysql",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
    )
    return conn

# Create SQLAlchemy engine using the creator pattern (no credentials in URL)
engine: Engine = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn, future=True)

# Context manager helpers to ensure connector closed and errors captured
from contextlib import contextmanager

@contextmanager
def safe_connection(engine):
    """
    Yields a connection and ensures proper cleanup.
    Use as:
      with safe_connection(engine) as conn:
          conn.execute(...)
    """
    conn = None
    try:
        conn = engine.connect()
        yield conn
    finally:
        if conn:
            conn.close()

"""Schema exploration helpers."""

# Schema exploration functions
import pandas as pd

def list_tables(engine) -> list:
    """Return a list of table names"""
    with safe_connection(engine) as conn:
        result = conn.execute(text("SHOW TABLES;")).fetchall()
    return [r[0] for r in result]

def get_table_columns(engine, table_name: str) -> pd.DataFrame:
    """Return a DataFrame of columns"""
    query = text("""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
    """)
    with safe_connection(engine) as conn:
        df = pd.read_sql(query, conn, params={"db": DB_NAME, "table": table_name})
    return df

"""Smoke tests."""
def fetch_sample_customers(limit: int = 10):
    q = text("SELECT customerNumber, customerName, country FROM customers LIMIT :limit;")
    with safe_connection(engine) as conn:
        df = pd.read_sql(q, conn, params={"limit": limit})
    return df

# Quick smoke test
try:
    tables = list_tables(engine)
    logger.info("Tables in classicmodels: %s", tables)
    sample_df = fetch_sample_customers(5)
    display(sample_df)
except Exception as e:
    logger.exception("Smoke test failed: %s", e)
finally:
    # leave connector open for notebook lifecycle, but show how to close
    # connector.close()  # uncomment to close when finished
    pass

"""# Justification for QueryRunner Class

**Purpose**:
The QueryRunner class is a central component for safely and systematically executing SQL queries within this NL-to-SQL evaluation framework. Its design incorporates several key features:

**Safety Checks**(_safety_check):

Crucially, it includes logic to prevent the execution of potentially destructive SQL commands (DROP, DELETE, TRUNCATE, ALTER, CREATE).

**History Tracking:** Every executed query, along with its outcome (success/failure, execution time, row count, errors, and a result preview)

**Result Capture:** Query results are captured into pandas.DataFrame objects, making them easy to analyze, display, and further process within the Python environment.

**Error Handling:** Robust try...except blocks ensure that database errors are caught, logged, and associated with the specific query in the history.

**Reproducibility and Evaluation:** By logging comprehensive metadata for each query, the QueryRunner directly supports the evaluation of NL-to-SQL models, enabling analysis of generated SQL correctness and execution behavior.


**Sources**: sqlalchemy for database interaction, pandas for data handling, Python's datetime and json for timestamping and serialization, software engineering principles for robust and safe code.
"""

# QueryRunner with timezone-aware datetimes
import pandas as pd
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

class QueryExecutionError(Exception):
    pass

def now_utc_iso() -> str:
    """Return current UTC time as ISO8601 string with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class QueryRunner:
    """
    Execute generated SQL safely against the engine, capture results and metadata,
    and keep a history suitable for evaluation and error analysis.
    """
    def __init__(self, engine, max_rows: int = 1000, forbidden_tokens=None):
        self.engine = engine
        self.max_rows = max_rows
        self.history = []  # list of dicts, append-only
        # Default to read-only by blocking common write/destructive verbs
        self.forbidden_tokens = forbidden_tokens or ["drop ", "delete ", "truncate ", "alter ", "create ", "update ", "insert "]

    def _safety_check(self, sql: str) -> None:
        lowered = (sql or "").strip().lower()
        if not lowered:
            raise QueryExecutionError("Empty SQL string")
        for token in self.forbidden_tokens:
            if token in lowered:
                raise QueryExecutionError(f"Destructive SQL token detected: {token.strip()}")

    def run(self, sql: str, params: Optional[Dict[str, Any]] = None, capture_df: bool = True) -> Dict[str, Any]:
        """
        Execute a read-only SQL statement and return a metadata dict with
         sql, params, timestamp, success, rowcount, exec_time_s, error, columns, result_preview
        """
        entry = {
            "sql": sql,
            "params": params,
            "timestamp": now_utc_iso(),
            "success": False,
            "rowcount": 0,
            "exec_time_s": None,
            "error": None,
            "columns": None,
            "result_preview": None,
        }

        try:
            self._safety_check(sql)
            start = datetime.now(timezone.utc)
            with safe_connection(self.engine) as conn:
                result = conn.execute(sqlalchemy.text(sql), params or {})
                rows = result.fetchall()
                cols = list(result.keys())
            end = datetime.now(timezone.utc)
            exec_time = (end - start).total_seconds()

            df = None
            if capture_df:
                df = pd.DataFrame(rows, columns=cols)
                if len(df) > self.max_rows:
                    df = df.iloc[: self.max_rows]

            entry.update({
                "success": True,
                "rowcount": min(len(rows), self.max_rows),
                "exec_time_s": exec_time,
                "columns": cols,
                "result_preview": df
            })

        except Exception as e:
            entry.update({
                "error": str(e),
                "success": False
            })
        finally:
            self.history.append(entry)
        return entry

    def last(self) -> Optional[Dict[str, Any]]:
        return self.history[-1] if self.history else None

    def save_history(self, path: str):
        """
        Save serializable parts of history to JSON. DataFrames are not serialized.
        """
        serializable = []
        for h in self.history:
            s = {k: v for k, v in h.items() if k != "result_preview"}
            serializable.append(s)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)

"""Quick QueryRunner smoke test."""

qr = QueryRunner(engine, max_rows=200)
test_sql = "SELECT customerNumber, customerName, country FROM customers LIMIT 10;"
meta = qr.run(test_sql)
print("Success:", meta["success"])
if meta["success"]:
    display(meta["result_preview"])
else:
    print("Error:", meta["error"])

tables = list_tables(engine)
logger.info("Tables in classicmodels: %s", tables)
print(f"Tables in classicmodels: {tables}")

for table_name in tables:
    print(f"\nSchema for table: {table_name}")
    df_columns = get_table_columns(engine, table_name)
    display(df_columns)

"""Dataset validation helper: run the static test set against live DB."""

def validate_test_set(path: str = "data/classicmodels_test_200.json", limit: Optional[int] = None):
    """
    Run the NLQ->SQL pairs in the given JSON file using QueryRunner.
    Args:
        path: location of the test set JSON.
        limit: optional cap on how many to run (for quick smoke tests).
    Prints a summary and returns (successes, failures).
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if limit:
        items = items[:limit]

    qr = QueryRunner(engine, max_rows=200)
    successes = []
    failures = []
    for idx, item in enumerate(items):
        meta = qr.run(item["sql"], capture_df=False)
        if meta["success"]:
            successes.append(idx)
        else:
            failures.append({
                "index": idx,
                "nlq": item.get("nlq"),
                "sql": item.get("sql"),
                "error": meta["error"],
            })
    print(f"Ran {len(items)} queries. Success: {len(successes)}. Failures: {len(failures)}.")
    if failures:
        print("Failures (first 5):")
        for f in failures[:5]:
            print(f)
    return successes, failures

"""Data prep: tiny starter set for quick checks (main test set lives in data/)."""

import json

nlq_sql_pairs = [
    {"nlq": "What are all the product lines available in our catalog?", "sql": "SELECT productLine FROM productlines;"},
    {"nlq": "List all products, their product codes, and their MSRPs.", "sql": "SELECT productName, productCode, MSRP FROM products;"},
    {"nlq": "Show the names of customers from the USA.", "sql": "SELECT customerName FROM customers WHERE country = 'USA';"},
    {"nlq": "How many employees work in the 'San Francisco' office?", "sql": "SELECT COUNT(*) FROM employees e JOIN offices o ON e.officeCode = o.officeCode WHERE o.city = 'San Francisco';"},
    {"nlq": "Find the total amount for each order number.", "sql": "SELECT orderNumber, SUM(quantityOrdered * priceEach) AS total_amount FROM orderdetails GROUP BY orderNumber;"},
    {"nlq": "Which customers have a credit limit greater than 100000?", "sql": "SELECT customerName FROM customers WHERE creditLimit > 100000;"},
    {"nlq": "List all offices and their locations (city, country).", "sql": "SELECT city, country FROM offices;"},
    {"nlq": "What are the details of payments made by customer number 103?", "sql": "SELECT checkNumber, paymentDate, amount FROM payments WHERE customerNumber = 103;"},
    {"nlq": "Show the product names and their vendors for products in the 'Classic Cars' product line.", "sql": "SELECT productName, productVendor FROM products WHERE productLine = 'Classic Cars';"},
    {"nlq": "How many orders have a 'Shipped' status?", "sql": "SELECT COUNT(*) FROM orders WHERE status = 'Shipped';"}
]

# Save the NLQ-SQL pairs to a JSON file
file_path = "curated_nlq_sql_pairs.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nlq_sql_pairs, f, indent=4)

print(f"NLQ-SQL pairs saved to {file_path}")

"""Load base model/tokenizer (pre-QLoRA placeholder)."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=True
)

print(f"Tokenizer and model '{model_id}' loaded successfully.")

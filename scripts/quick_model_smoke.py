#!/usr/bin/env python3
"""
Quick, DB-free model sanity check for NL->SQL generation.

Loads the base model + optional QLoRA adapter and runs a few NLQs against
an in-memory schema string. Prints raw output, cleaned SQL, and a lightweight
schema/format validation result (no execution).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from nl2sql.llm import generate_sql_from_messages
from nl2sql.prompting import SYSTEM_INSTRUCTIONS
from nl2sql.agent_utils import clean_candidate_with_reason
from nl2sql.agent_tools import validate_sql


DEFAULT_SCHEMA = """\
customers(customerNumber, customerName, contactLastName, contactFirstName, country)
orders(orderNumber, orderDate, status, customerNumber)
orderdetails(orderNumber, productCode, quantityOrdered, priceEach)
products(productCode, productName, productLine, buyPrice)
"""

DEFAULT_NLQS = [
    "List customer names and countries for customers in France.",
    "Show the total number of orders per customer.",
    "Find product names and prices for products in the 'Classic Cars' line.",
]


def _load_model(model_id: str, adapter_path: str | None, token: str, max_new_tokens: int):
    has_cuda = torch.cuda.is_available()
    cc_major, _ = torch.cuda.get_device_capability(0) if has_cuda else (0, 0)
    use_bf16 = has_cuda and cc_major >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print("GPU:", torch.cuda.get_device_name(0) if has_cuda else "CPU")
    print("Using bf16:", use_bf16)

    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map={"": 0},
            token=token,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
            token=token,
        )
    base_model.generation_config.do_sample = False
    base_model.generation_config.temperature = 1.0
    base_model.generation_config.top_p = 1.0

    model = base_model
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists():
            model = PeftModel.from_pretrained(base_model, adapter_dir, token=token)
            print("Loaded adapters from", adapter_dir)
        else:
            print("Adapter path missing; using base model only:", adapter_dir)
    else:
        print("No adapter path provided; using base model only.")

    return model, tok


def _gen_sql(model, tok, schema_text: str, nlq: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float):
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": f"Schema:\\n{schema_text}"},
        {"role": "user", "content": f"NLQ: {nlq}\\nReturn a single SQL SELECT."},
    ]
    return generate_sql_from_messages(
        model=model,
        tokenizer=tok,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DB-free NL->SQL model smoke test.")
    parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter", default=os.getenv("ADAPTER_PATH"))
    parser.add_argument("--schema-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("Missing HF_TOKEN/HUGGINGFACE_TOKEN env var.")

    schema_text = DEFAULT_SCHEMA
    if args.schema_file:
        schema_text = Path(args.schema_file).read_text(encoding="utf-8")

    model, tok = _load_model(args.model_id, args.adapter, token, args.max_new_tokens)

    print("\n=== Smoke test ===")
    for idx, nlq in enumerate(DEFAULT_NLQS, start=1):
        raw = _gen_sql(
            model,
            tok,
            schema_text,
            nlq,
            args.max_new_tokens,
            args.do_sample,
            args.temperature,
            args.top_p,
        )
        cleaned, reason = clean_candidate_with_reason(raw)
        val = validate_sql(cleaned or raw, schema_text)
        print(f"\n[{idx}] NLQ: {nlq}")
        print("RAW:", raw)
        print("CLEAN:", cleaned or "")
        print("CLEAN_REASON:", reason)
        print("VALIDATE:", val)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

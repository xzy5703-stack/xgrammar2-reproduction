"""Paired greedy Qwen3 tool-call experiment with the official HF XGrammar adapter.

This is a local experiment on a fixed BFCL-v3 prefix, not the official SGLang
BFCL score. JSON-schema compliance and argument accuracy are distinct metrics.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import time
from pathlib import Path

import jsonschema
import numpy as np
import torch
import xgrammar as xg
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xgrammar.contrib.hf import LogitsProcessor

ROOT = Path(__file__).resolve().parent


def strict_json_loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("non-finite JSON constant: " + value)
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("overflow in JSON number")
        return parsed
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)


def normalize_schema(value):
    """Translate BFCL Python type spellings to JSON Schema; preserve constraints."""
    mapping = {"dict": "object", "list": "array", "tuple": "array", "float": "number",
               "str": "string", "int": "integer", "bool": "boolean"}
    if isinstance(value, list):
        return [normalize_schema(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {k: normalize_schema(v) for k, v in value.items()}
    if "type" in result:
        original = result["type"]
        if original == "any":
            del result["type"]
        elif isinstance(original, str):
            result["type"] = mapping.get(original, original)
    if result.get("type") == "object":
        result.setdefault("additionalProperties", False)
    return result


def call_schema(task):
    variants = []
    for function in task["function"]:
        parameters = normalize_schema(function["parameters"])
        jsonschema.Draft202012Validator.check_schema(parameters)
        variants.append({"type": "object", "properties": {
            "name": {"type": "string", "const": function["name"]},
            "arguments": parameters}, "required": ["name", "arguments"],
            "additionalProperties": False})
    return variants[0] if len(variants) == 1 else {"anyOf": variants}


def values_equal(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(values_equal(x, y) for x, y in zip(actual, expected))
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(values_equal(actual[k], expected[k]) for k in actual)
    return actual == expected


def score_arguments(call, answer, task):
    """Local acceptable-value checker, without execution or fuzzy matching.

    Outer answer lists denote alternative accepted values, not array arguments.
    A missing optional argument is accepted only if the reference includes "".
    """
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        return False
    reference_calls = answer.get("ground_truth", [])
    if not isinstance(reference_calls, list) or len(reference_calls) != 1:
        raise ValueError("BFCL simple single-call reference required")
    expected = reference_calls[0]
    name = call.get("name")
    if name not in expected:
        return False
    function = next((x for x in task["function"] if x["name"] == name), None)
    if function is None:
        return False
    required = function["parameters"].get("required", [])
    allowed = expected[name]
    actual = call["arguments"]
    if set(actual) - set(allowed):
        return False
    for key, alternatives in allowed.items():
        if key not in actual:
            if key in required or "" not in alternatives:
                return False
        elif not any(values_equal(actual[key], item) for item in alternatives):
            return False
    return True


class TimedProcessor(LogitsProcessor):
    def __init__(self, grammar):
        super().__init__(grammar)
        self.latencies = []

    def __call__(self, input_ids, scores):
        if scores.device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter_ns()
        # First call uses the author's initialization and masking directly.
        # Subsequent calls follow the same batch-size-one state transitions,
        # passing a Python int (not a torch scalar) to the documented matcher
        # API. Current TVM FFI rejects both CPU and MPS torch scalar IDs.
        if not self.matchers:
            output = super().__call__(input_ids.to("cpu"), scores)
        else:
            assert input_ids.shape[0] == 1 and len(self.matchers) == 1
            matcher = self.matchers[0]
            if not matcher.is_terminated():
                assert matcher.accept_token(int(input_ids[0, -1].item()))
                matcher.fill_next_token_bitmask(self.token_bitmask, 0)
            cpu_scores = scores.to("cpu")
            xg.apply_token_bitmask_inplace(cpu_scores, self.token_bitmask)
            output = cpu_scores.to(scores.device)
        if scores.device.type == "mps":
            torch.mps.synchronize()
        self.latencies.append((time.perf_counter_ns() - start) / 1000)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--out", default="results")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    task_path = ROOT / "inputs" / "bfcl_simple.jsonl"
    answer_path = ROOT / "inputs" / "bfcl_simple_answer.jsonl"
    tasks = [json.loads(line) for line in task_path.read_text().splitlines() if line][:args.count]
    answers = {x["id"]: x for x in map(json.loads, answer_path.read_text().splitlines())}
    assert len(tasks) == args.count and all(x["id"] in answers for x in tasks)
    torch.manual_seed(20261015)
    torch.set_num_threads(6)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model_path = ROOT / "models" / "Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
             trust_remote_code=False, dtype=torch.float16 if device == "mps" else torch.float32)
    model.to(device).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    tokenizer_info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    compiler = xg.GrammarCompiler(tokenizer_info, max_threads=1)
    print(f"Model loaded: Qwen3-0.6B on {device}; paired cases={len(tasks)}", flush=True)
    # One short prefill/decode warmup; not a scored benchmark case.
    with torch.inference_mode():
        warm = tokenizer("Hello", return_tensors="pt").to(device)
        model.generate(**warm, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    rows = []
    for index, task in enumerate(tasks):
        schema = call_schema(task)
        messages = [{"role": "system", "content":
            "Choose the supplied function and its arguments for the user's request. "
            "Respond with exactly one JSON object containing name and arguments. "
            "Do not explain, calculate the function result, or use markdown. "
            "Available functions:\n" + json.dumps(task["function"], ensure_ascii=False)},
            *task["question"][0]]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Alternate paired ordering to reduce systematic warmup/order bias.
        modes = ["unconstrained", "xgrammar"] if index % 2 == 0 else ["xgrammar", "unconstrained"]
        for mode in modes:
            t0 = time.perf_counter_ns()
            grammar = compiler.compile_json_schema(schema) if mode == "xgrammar" else None
            compile_ms = (time.perf_counter_ns() - t0) / 1e6 if grammar is not None else 0.0
            processor = TimedProcessor(grammar) if grammar is not None else None
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, logits_processor=[processor] if processor else [],
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id, use_cache=True)
            if device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - t0
            token_ids = generated[0, inputs["input_ids"].shape[1]:].cpu().tolist()
            completed = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            parsed = None
            json_valid = schema_valid = False
            error = ""
            try:
                parsed = strict_json_loads(text)
                json_valid = True
                jsonschema.Draft202012Validator(schema).validate(parsed)
                schema_valid = True
            except (ValueError, jsonschema.ValidationError) as exc:
                error = type(exc).__name__
            semantic = schema_valid and completed and score_arguments(parsed, answers[task["id"]], task)
            # Supplementary content diagnostic: strip only a single whole-output
            # Markdown JSON fence. This never changes strict compliance/success.
            fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, flags=re.DOTALL)
            content_text = fenced.group(1) if fenced else text
            content_exact = False
            try:
                content_call = strict_json_loads(content_text)
                jsonschema.Draft202012Validator(schema).validate(content_call)
                content_exact = completed and score_arguments(content_call, answers[task["id"]], task)
            except (ValueError, jsonschema.ValidationError):
                pass
            row = {"id": task["id"], "mode": mode, "output": text, "tokens": token_ids,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_token_count": inputs["input_ids"].shape[1],
                "prompt_tokens_sha256": hashlib.sha256(json.dumps(inputs["input_ids"][0].cpu().tolist()).encode()).hexdigest(),
                "completed": completed, "json_valid": json_valid, "schema_valid": schema_valid,
                "argument_exact": bool(semantic), "error": error, "wall_seconds": elapsed,
                "content_argument_exact": bool(content_exact), "whole_markdown_fence": bool(fenced),
                "compile_ms": compile_ms, "processor_us": processor.latencies if processor else [],
                "schema_sha256": hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()}
            rows.append(row)
            (out / "generation_records.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
            print(f"{index+1}/{len(tasks)} {task['id']} {mode}: complete={completed} schema={schema_valid} "
                  f"argument={semantic} tokens={len(token_ids)} seconds={elapsed:.2f}", flush=True)
    metrics = {}
    for mode in ["unconstrained", "xgrammar"]:
        selected = [x for x in rows if x["mode"] == mode]
        metrics[mode] = {"n": len(selected), **{k: sum(x[k] for x in selected)
            for k in ["completed", "json_valid", "schema_valid", "argument_exact", "content_argument_exact"]},
            "total_tokens": sum(len(x["tokens"]) for x in selected),
            "total_seconds": sum(x["wall_seconds"] for x in selected)}
    info = {"experiment": "paired_local_json_tool_call", "official_bfcl_score": False,
        "sample_rule": "first N records of pinned BFCL-v3 simple file, no outcome filtering",
        "ids": [x["id"] for x in tasks], "max_new_tokens": args.max_new_tokens,
        "seed": 20261015, "do_sample": False, "thinking": False, "device": device,
        "dtype": str(model.dtype), "platform": platform.platform(),
        "packages": {x: importlib.metadata.version(x) for x in ["torch", "xgrammar", "transformers", "jsonschema"]},
        "hf_adapter_change": "Official v0.2.0 HF initialization; subsequent batch-size-one transitions use Python int token ID, CPU masks and MPS scores. Author matcher and mask kernels unchanged; current TVM FFI rejects torch scalar IDs.",
        "metrics": metrics, "timing_scope": "includes CPU mask and MPS transfers; not serving throughput"}
    info["metric_definitions"] = {
        "argument_exact": "Strict complete JSON-schema-compliant tool call with all acceptable reference argument values; invalid format counts as failure",
        "content_argument_exact": "Supplementary semantic diagnostic; permits stripping only a single whole-output Markdown JSON fence. This is not strict format compliance or an official BFCL score",
        "schema_valid": "Strict original output is JSON and passes independent JSON Schema; completion reported separately"}
    (out / "generation_summary.json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(metrics,indent=2), flush=True)


if __name__ == "__main__":
    main()

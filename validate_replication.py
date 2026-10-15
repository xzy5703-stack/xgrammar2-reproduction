#!/usr/bin/env python3
"""Independent offline audit; neither experiment runner is imported.

Missing or incomplete experiment artifacts fail. --self-test-only validates the
checker helpers and must not be reported as successful replication verification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE_COMMIT = "5433fdbea7003eea61cb863768a05fb8da3bf919"
MODEL_COMMIT = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_COMMIT = "cd9429ccf3d4d04156affe883c495b3b047e6b64"
SOURCE_ARCHIVE_SHA256 = "b5fc16b0cfedb9350b6529c058c1b736230817804649d977b05a09550a9a40e3"
MODEL_WEIGHT_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
DATA_SHA256 = {
    "bfcl_simple.jsonl": "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
    "bfcl_simple_answer.jsonl": "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee"}
ALIASES = {"dict": "object", "list": "array", "tuple": "array", "float": "number",
           "str": "string", "int": "integer", "bool": "boolean",
           "None": "null", "NoneType": "null"}
SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
SCHEMA_CHILDREN = {"items", "additionalProperties", "contains", "propertyNames", "if", "then",
                   "else", "not", "unevaluatedProperties", "unevaluatedItems", "additionalItems",
                   "allOf", "anyOf", "oneOf", "prefixItems"}
POLICIES = ["disabled", "persistent", "reset_each_request"]


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("non-finite JSON constant " + value)

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("number outside finite float range")
        return number

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float)


def read_json(path):
    return strict_json(path.read_text())


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines() if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalize(schema, *, close_objects):
    """Visit schema positions only; enum/default/example payloads remain data."""
    if isinstance(schema, list):
        return [normalize(x, close_objects=close_objects) for x in schema]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        if key in SCHEMA_MAPS:
            result[key] = {name: normalize(child, close_objects=close_objects)
                           for name, child in value.items()}
        elif key in SCHEMA_CHILDREN:
            result[key] = normalize(value, close_objects=close_objects)
        elif key == "type":
            def convert(item):
                if item in {"any", "Any"}:
                    return None if close_objects else "string"
                return ALIASES.get(item, item)
            converted = [convert(x) for x in value] if isinstance(value, list) else convert(value)
            if converted is not None and not (isinstance(converted, list) and None in converted):
                result[key] = converted
        else:
            result[key] = value
    if close_objects and result.get("type") == "object":
        result.setdefault("additionalProperties", False)
    return result


def call_schema(task):
    alternatives = [
        {"type": "object", "properties": {
            "name": {"type": "string", "const": f["name"]},
            "arguments": normalize(f["parameters"], close_objects=True)},
         "required": ["name", "arguments"], "additionalProperties": False}
        for f in task["function"]]
    return alternatives[0] if len(alternatives) == 1 else {"anyOf": alternatives}


def equal_values(left, right):
    """Numeric int/float equality, with recursively type-strict boolean handling."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        finite_left = not isinstance(left, float) or math.isfinite(left)
        finite_right = not isinstance(right, float) or math.isfinite(right)
        return finite_left and finite_right and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(equal_values(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_values(left[k], right[k]) for k in left)
    return left == right


def reference_matches(call, answer, task):
    if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
        return False
    reference = answer["ground_truth"]
    if not isinstance(reference, list) or len(reference) != 1:
        raise ValueError("one reference function call required for BFCL simple")
    acceptable_calls = reference[0]
    name = call.get("name")
    functions = {f["name"]: f for f in task["function"]}
    if name not in functions or name not in acceptable_calls:
        return False
    arguments, acceptable = call["arguments"], acceptable_calls[name]
    required = set(functions[name]["parameters"].get("required", []))
    if not required <= arguments.keys() or not arguments.keys() <= acceptable.keys():
        return False
    for parameter, choices in acceptable.items():
        if not isinstance(choices, list):
            raise ValueError("reference alternatives must be lists")
        if parameter in arguments:
            if not any(equal_values(arguments[parameter], item) for item in choices):
                return False
        elif parameter in required or not any(type(x) is str and x == "" for x in choices):
            return False
    return True


class Audit:
    def __init__(self):
        self.sections = {}
        self.section = ""

    def check(self, condition, message):
        section = self.sections[self.section]
        section["checks"] += 1
        if bool(condition):
            section["passed"] += 1
        else:
            section["failures"].append(message)

    def equal(self, actual, expected, message):
        self.check(actual == expected, f"{message}: got={str(actual)[:200]}, expected={str(expected)[:200]}")

    def close(self, actual, expected, message):
        self.check(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9),
                   f"{message}: got={actual}, recomputed={expected}")

    def stage(self, name, callback):
        self.section = name
        self.sections[name] = {"checks": 0, "passed": 0, "failures": []}
        try:
            callback()
        except Exception as exc:
            self.check(False, f"{type(exc).__name__}: {exc}")


def self_tests(audit):
    for text in ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}']:
        try:
            strict_json(text)
            rejected = False
        except ValueError:
            rejected = True
        audit.check(rejected, "reject invalid JSON: " + text)
    audit.equal(strict_json(' {"a":[1,true,null]} '), {"a": [1, True, None]}, "parse valid JSON")
    for left, right, expected in [(1, 1.0, True), (True, 1, False), ([True], [1], False),
                                 ({"x": [False]}, {"x": [0]}, False), ([1, 2], [1.0, 2.0], True),
                                 ("12", 12, False), ([1], [1, 2], False), (10 ** 500, 10 ** 500, True)]:
        audit.equal(equal_values(left, right), expected, "recursive strict reference equality")
    task = {"function": [{"name": "f", "parameters": {"required": ["x"]}}]}
    answer = {"ground_truth": [{"f": {"x": [1, ""], "optional": [""]}}]}
    audit.check(reference_matches({"name": "f", "arguments": {"x": 1}}, answer, task), "optional omission allowed")
    audit.check(not reference_matches({"name": "f", "arguments": {}}, answer, task), "required omission rejected")
    audit.check(not reference_matches({"name": "f", "arguments": {"x": True}}, answer, task), "boolean is not numeric answer")
    audit.check(not reference_matches({"name": "f", "arguments": {"x": 1, "extra": 2}}, answer, task), "extra argument rejected")
    payload = {"type": "dict", "properties": {"x": {"enum": [{"type": "dict"}]}}}
    audit.equal(normalize(payload, close_objects=True)["properties"]["x"]["enum"],
                [{"type": "dict"}], "enum data is not converted as a schema")


def provenance(audit, root):
    source = read_json(root / "source_manifest.json")
    model = read_json(root / "model_manifest.json")
    data = read_json(root / "inputs/data_provenance.json")
    for key, expected in [("engine_version", "0.2.0"), ("engine_revision", ENGINE_COMMIT),
                          ("model_revision", MODEL_COMMIT)]:
        audit.equal(source[key], expected, "fixed " + key)
    audit.equal(model["revision"], MODEL_COMMIT, "model manifest fixed revision")
    audit.equal(source["model_repo"], "Qwen/Qwen3-0.6B", "fixed source model repository")
    audit.equal(model["repo"], "Qwen/Qwen3-0.6B", "fixed model manifest repository")
    audit.equal(data["source_revision"], DATA_COMMIT, "BFCL fixed revision")
    audit.equal(importlib.metadata.version("xgrammar"), "0.2.0", "installed official engine release")
    archive = root / "inputs/xgrammar-v0.2.0.zip"
    audit.equal(source["engine_archive_sha256"], SOURCE_ARCHIVE_SHA256, "predeclared author archive hash")
    audit.equal(file_digest(archive), source["engine_archive_sha256"], "source archive SHA256")
    installed_package = Path(importlib.util.find_spec("xgrammar").origin).parent
    compared, installed_compared = 0, 0
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if entry.is_dir():
                continue
            relative = Path(entry.filename)
            safe = not relative.is_absolute() and ".." not in relative.parts
            audit.check(safe, "safe source archive path")
            if not safe:
                raise ValueError("unsafe source archive path")
            audit.equal(file_digest(root / "vendor" / relative), hashlib.sha256(bundle.read(entry)).hexdigest(),
                        "unmodified author source " + str(relative))
            if len(relative.parts) > 3 and relative.parts[1:3] == ("python", "xgrammar") and relative.suffix == ".py":
                installed = installed_package.joinpath(*relative.parts[3:])
                audit.equal(file_digest(installed), hashlib.sha256(bundle.read(entry)).hexdigest(),
                            "installed official Python API matches pinned source " + str(Path(*relative.parts[3:])))
                installed_compared += 1
            compared += 1
    audit.check(compared > 50, "complete author archive inspected")
    audit.check(installed_compared > 10, "installed author Python API inspected")
    for name, metadata in model["assets"].items():
        if Path(name).name != name:
            raise ValueError("unsafe model asset path")
        path = root / "models/Qwen3-0.6B" / name
        audit.equal(path.stat().st_size, metadata["bytes"], "model asset bytes " + name)
        audit.equal(file_digest(path), metadata["sha256"], "model asset SHA256 " + name)
    audit.equal(model["assets"]["model.safetensors"]["sha256"], MODEL_WEIGHT_SHA256, "predeclared model weight hash")
    datasets = {}
    for item in data["files"]:
        if item["local_path"] not in {"inputs/bfcl_simple.jsonl", "inputs/bfcl_simple_answer.jsonl"}:
            raise ValueError("undeclared input data path")
        path = root / item["local_path"]
        raw = path.read_bytes()
        audit.equal(len(raw), item["bytes"], "dataset bytes " + path.name)
        audit.equal(hashlib.sha256(raw).hexdigest(), item["sha256"], "dataset SHA256 " + path.name)
        audit.equal(item["sha256"], DATA_SHA256[path.name], "predeclared dataset hash " + path.name)
        audit.equal(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
                    item["github_blob_sha1"], "GitHub blob identity " + path.name)
        rows = read_jsonl(path)
        audit.equal(len(rows), item["rows"], "dataset row count " + path.name)
        ids = [row["id"] for row in rows]
        audit.equal(len(set(ids)), len(ids), "unique original IDs " + path.name)
        datasets[path.name] = ids
    audit.equal(datasets["bfcl_simple.jsonl"], datasets["bfcl_simple_answer.jsonl"], "question/answer ID alignment")
    audit.equal(datasets["bfcl_simple.jsonl"][:32], [f"simple_{i}" for i in range(32)], "predeclared source prefix")


def generation(audit, root, out):
    import jsonschema
    import xgrammar as xgr
    from transformers import AutoTokenizer
    tasks = read_jsonl(root / "inputs/bfcl_simple.jsonl")[:32]
    answers = {row["id"]: row for row in read_jsonl(root / "inputs/bfcl_simple_answer.jsonl")}
    rows = read_jsonl(out / "generation_records.jsonl")
    summary = read_json(out / "generation_summary.json")
    audit.equal(set(summary["metrics"]), {"unconstrained", "xgrammar"}, "only declared generation mode summaries")
    audit.equal(len(rows), 64, "complete 32-case paired generation")
    expected = Counter((task["id"], mode) for task in tasks for mode in ("unconstrained", "xgrammar"))
    audit.equal(Counter((row["id"], row["mode"]) for row in rows), expected, "each predeclared ID once per mode")
    expected_order = [(task["id"], mode) for i, task in enumerate(tasks)
                      for mode in (("unconstrained", "xgrammar") if i % 2 == 0 else ("xgrammar", "unconstrained"))]
    audit.equal([(row["id"], row["mode"]) for row in rows], expected_order, "balanced alternating model-mode order")
    audit.equal(summary["ids"], [task["id"] for task in tasks], "summary ID prefix")
    audit.equal(summary["seed"], 20261015, "fixed generation seed")
    audit.check(summary["do_sample"] is False and summary["thinking"] is False, "greedy no-thinking protocol")
    audit.check(summary["official_bfcl_score"] is False, "local metric is not official BFCL score")
    audit.equal(summary["packages"]["xgrammar"], "0.2.0", "recorded engine version")
    for package, version in summary["packages"].items():
        audit.equal(importlib.metadata.version(package), version, "current package matches generation " + package)
    audit.check("Python int token ID" in summary["hf_adapter_change"] and
                "matcher and mask kernels unchanged" in summary["hf_adapter_change"],
                "MPS/FFI adapter change is explicitly recorded")
    tokenizer = AutoTokenizer.from_pretrained(root / "models/Qwen3-0.6B", local_files_only=True, trust_remote_code=False)
    model_config = read_json(root / "models/Qwen3-0.6B/config.json")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model_config["vocab_size"])
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
    task_lookup = {task["id"]: task for task in tasks}
    totals = defaultdict(list)
    fence_pattern = re.compile(r"\A\s*" + chr(96) * 3 + r"(?:json)?\s*\n(.*?)\n" + chr(96) * 3 + r"\s*\Z", re.DOTALL)
    for row in rows:
        label = row["id"] + "/" + row["mode"]
        task = task_lookup[row["id"]]
        schema = call_schema(task)
        jsonschema.Draft202012Validator.check_schema(schema)
        audit.equal(row["schema_sha256"], hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                    label + " independent full call schema")
        tokens = row["tokens"]
        audit.check(bool(tokens) and all(type(t) is int and 0 <= t < len(tokenizer) for t in tokens), label + " emitted token IDs")
        audit.equal(tokenizer.decode(tokens, skip_special_tokens=True), row["output"], label + " decoded output")
        completed = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
        audit.equal(row["completed"], completed, label + " EOS completion")
        audit.check(len(tokens) <= summary["max_new_tokens"], label + " token budget")
        audit.check(completed or len(tokens) == summary["max_new_tokens"], label + " EOS or maximum-token truncation")
        audit.check(tokenizer.eos_token_id not in tokens[:-1], label + " no earlier EOS hidden by decode")
        parsed, json_valid, schema_valid = None, False, False
        try:
            parsed = strict_json(row["output"])
            json_valid = True
            schema_valid = jsonschema.Draft202012Validator(schema).is_valid(parsed)
        except ValueError:
            pass
        semantic = completed and schema_valid and reference_matches(parsed, answers[row["id"]], task)
        match = fence_pattern.fullmatch(row["output"])
        content_semantic = semantic
        if match:
            try:
                content = strict_json(match.group(1))
                content_semantic = completed and jsonschema.Draft202012Validator(schema).is_valid(content) and reference_matches(
                    content, answers[row["id"]], task)
            except ValueError:
                content_semantic = False
        for key, expected_value in {"json_valid": json_valid, "schema_valid": schema_valid, "argument_exact": semantic,
                                    "whole_markdown_fence": bool(match), "content_argument_exact": content_semantic}.items():
            audit.equal(row[key], expected_value, label + " independently recomputed " + key)
        audit.check(math.isfinite(row["wall_seconds"]) and row["wall_seconds"] > 0, label + " measured positive duration")
        audit.check(math.isfinite(row["compile_ms"]) and row["compile_ms"] >= 0, label + " finite compile duration")
        latencies = row["processor_us"]
        audit.check(all(math.isfinite(x) and x > 0 for x in latencies), label + " finite processor timings")
        audit.equal(len(latencies), len(tokens) if row["mode"] == "xgrammar" else 0, label + " processor calls per token")
        if row["mode"] == "unconstrained":
            audit.equal(row["compile_ms"], 0.0, label + " unconstrained compilation absent")
        else:
            # Replay the actual saved sequence without importing the runner or
            # its adapter. Independently unpack mask bits at every decode step.
            matcher = xgr.GrammarMatcher(grammar_compiler.compile_json_schema(schema))
            mask = xgr.allocate_token_bitmask(1, model_config["vocab_size"])
            allowed_sequence, mask_consistent = True, True
            for token in tokens:
                apply_mask = matcher.fill_next_token_bitmask(mask)
                word = int(mask[0, token // 32].item()) & 0xFFFFFFFF
                allowed = not apply_mask or bool((word >> (token % 32)) & 1)
                accepted = matcher.accept_token(int(token))
                allowed_sequence = allowed_sequence and allowed and accepted
                mask_consistent = mask_consistent and allowed == accepted
                if not accepted:
                    break
            audit.check(allowed_sequence, label + " independently replayed tokens all admitted by author grammar")
            audit.check(mask_consistent, label + " mask bits agree with author matcher transition")
            audit.equal(matcher.is_terminated(), completed, label + " author matcher termination equals EOS")
        messages = [{"role": "system", "content":
            "Choose the supplied function and its arguments for the user's request. "
            "Respond with exactly one JSON object containing name and arguments. "
            "Do not explain, calculate the function result, or use markdown. "
            "Available functions:\n" + json.dumps(task["function"], ensure_ascii=False)}, *task["question"][0]]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        audit.equal(row["prompt_sha256"], hashlib.sha256(prompt.encode()).hexdigest(), label + " original task prompt")
        prompt_tokens = tokenizer(prompt)["input_ids"]
        if "prompt_token_count" in row:
            audit.equal(row["prompt_token_count"], len(prompt_tokens), label + " prompt token count")
        if "prompt_tokens_sha256" in row:
            audit.equal(row["prompt_tokens_sha256"], hashlib.sha256(json.dumps(prompt_tokens).encode()).hexdigest(), label + " prompt token hash")
        totals[row["mode"]].append({"completed": completed, "json_valid": json_valid, "schema_valid": schema_valid,
            "argument_exact": semantic, "content_argument_exact": content_semantic, "tokens": len(tokens), "seconds": row["wall_seconds"]})
    for mode, records in totals.items():
        metric = summary["metrics"][mode]
        audit.equal(metric["n"], len(records), mode + " summary N")
        for key in ("completed", "json_valid", "schema_valid", "argument_exact", "content_argument_exact"):
            audit.equal(metric[key], sum(row[key] for row in records), mode + " independent metric total " + key)
        audit.equal(metric["total_tokens"], sum(row["tokens"] for row in records), mode + " total tokens")
        audit.close(metric["total_seconds"], sum(row["seconds"] for row in records), mode + " total seconds")


def boolean(value):
    if value not in {"True", "False"}:
        raise ValueError("invalid CSV boolean " + value)
    return value == "True"


def tools_from_original(root):
    distinct = {}
    for record in read_jsonl(root / "inputs/bfcl_simple.jsonl"):
        for function in record["function"]:
            tool = {"name": function["name"], "parameters": normalize(function["parameters"], close_objects=False)}
            distinct[digest(tool)] = tool
    return sorted(distinct.values(), key=canonical)


def grammar_for(tools):
    return {"type": "structural_tag", "format": {"type": "triggered_tags", "triggers": ["<function="],
        "tags": [{"type": "tag", "begin": "<function=" + tool["name"] + ">",
                  "content": {"type": "json_schema", "json_schema": tool["parameters"]}, "end": "</function>"}
                 for tool in sorted(tools, key=lambda item: item["name"])]}}


def benchmark(audit, root, out):
    summary, workloads = read_json(out / "summary.json"), read_json(out / "workload_manifest.json")
    with (out / "raw.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    audit.check(summary["smoke_run"] is False and summary["complete_repeat_protocol"] is True, "full benchmark rather than smoke")
    audit.check(summary["repeats"] >= 3, "at least three independent repeats")
    audit.equal(summary["seed"], 20261015, "predeclared workload seed")
    audit.equal(summary["executed_tool_counts"], [10, 50, 100, 200], "all predeclared tool counts executed")
    audit.equal(summary["requests_per_condition"], 20, "20 requests per condition")
    audit.equal(summary["max_compiler_threads"], 1, "compiler single-thread setting")
    audit.equal(summary["torch_threads"], 1, "torch single-thread setting")
    audit.equal(summary["xgrammar_version"], "0.2.0", "benchmark fixed engine")
    audit.equal(summary["author_source_commit"], ENGINE_COMMIT, "benchmark source commit")
    audit.equal(summary["input"]["input_sha256"], file_digest(root / "inputs/bfcl_simple.jsonl"), "benchmark original input")
    audit.equal(summary["input"]["skipped_records"], [], "no silently discarded input tools")
    tools = tools_from_original(root)
    lookup = {digest(tool): tool for tool in tools}
    variants_by_name = defaultdict(list)
    for index, tool in enumerate(tools):
        variants_by_name[tool["name"]].append(index)
    available_names = sorted(variants_by_name)
    audit.equal(len(tools), summary["input"]["distinct_tool_count"], "independently distinct original schemas")
    audit.equal(len({tool["name"] for tool in tools}), summary["input"]["distinct_tool_name_count"], "distinct names")
    repeats, requests, counts = summary["repeats"], summary["requests_per_condition"], summary["executed_tool_counts"]
    blocks = {}
    for block in workloads:
        key = (block["repeat"], block["mode"], block["tool_count"])
        audit.check(key not in blocks, "unique workload block " + str(key))
        blocks[key] = block
        audit.equal(block["seed"], summary["seed"] + block["repeat"] * 100003 + block["tool_count"], "independent repeat seed")
        audit.equal(len(block["requests"]), requests, "full block request count")
        # Recreate only the declared input selection, never the timed code.
        # This proves the logged seed corresponds to the original schema IDs.
        selection_rng = random.Random(block["seed"])
        generated_selections, seen_selections = [], set()
        while len(generated_selections) < (1 if block["mode"] == "static" else requests):
            indices = tuple(sorted(selection_rng.choice(variants_by_name[name])
                for name in selection_rng.sample(available_names, block["tool_count"])))
            if indices not in seen_selections:
                generated_selections.append(indices)
                seen_selections.add(indices)
        if block["mode"] == "static":
            generated_selections *= requests
        previous, prior_names, prior_parameters = set(), set(), set()
        for index, request in enumerate(block["requests"]):
            label = str(key) + "/" + str(index)
            selected = [lookup[identity] for identity in request["selected_tool_sha256"]]
            expected_tools = sorted((tools[i] for i in generated_selections[index]), key=lambda item: item["name"])
            audit.equal(request["selected_tool_sha256"], [digest(tool) for tool in expected_tools],
                        label + " independently recreated seeded original inputs")
            audit.equal(request["request_index"], index, label + " index")
            audit.equal(len(selected), block["tool_count"], label + " schema identity count")
            names = {tool["name"] for tool in selected}
            parameters = {digest(tool["parameters"]) for tool in selected}
            audit.equal(len(names), block["tool_count"], label + " distinct original names")
            audit.equal(sorted(names), request["tool_names"], label + " original schema identities/names")
            grammar = grammar_for(selected)
            grammar_hash = digest(grammar)
            audit.equal(request["grammar_sha256"], grammar_hash, label + " independently constructed author TagDispatch input")
            audit.equal(len(canonical(grammar).encode()), request["grammar_json_bytes"], label + " grammar byte length")
            audit.equal(request["whole_grammar_reused"], grammar_hash in previous, label + " recomputed whole-grammar reuse")
            audit.check((index == 0 or grammar_hash in previous) if block["mode"] == "static" else grammar_hash not in previous,
                        label + " static/dynamic design")
            audit.close(request["tool_overlap_with_previous_requests"], len(names & prior_names) / len(names), label + " name overlap")
            audit.close(request["parameter_structure_overlap_with_previous_requests"],
                        len(parameters & prior_parameters) / len(parameters), label + " original schema overlap")
            previous.add(grammar_hash)
            prior_names.update(names)
            prior_parameters.update(parameters)
        audit.equal(len(previous), 1 if block["mode"] == "static" else requests, "static/dynamic unique grammar count")
    expected_blocks = {(repeat, mode, count) for repeat in range(repeats) for mode in ("static", "dynamic") for count in counts}
    audit.equal(set(blocks), expected_blocks, "complete workload Cartesian product")
    audit.equal(len(rows), repeats * len(counts) * requests * 6, "complete three-control raw observations")
    pairs, conditions = defaultdict(list), defaultdict(list)
    for row in rows:
        repeat, count, index = int(row["repeat"]), int(row["tool_count"]), int(row["request_index"])
        key, policy = (repeat, row["mode"], count), row["cache_policy"]
        label = str((*key, index)) + "/" + policy
        metadata = blocks[key]["requests"][index]
        audit.check(policy in POLICIES, label + " declared control")
        audit.equal(boolean(row["cache_enabled"]), policy != "disabled", label + " engine cache setting")
        audit.equal(int(row["seed"]), blocks[key]["seed"], label + " seed")
        audit.equal(row["grammar_sha256"], metadata["grammar_sha256"], label + " identical paired input")
        audit.equal(row["error"], "", label + " compilation succeeded")
        audit.check(math.isfinite(float(row["compile_ms"])) and float(row["compile_ms"]) > 0, label + " positive finite compilation time")
        audit.check(int(row["compiled_grammar_bytes"]) > 0, label + " compiled grammar exists")
        audit.check(int(row["cache_bytes"]) >= 0 and int(row["cache_bytes_before_compile"]) >= 0, label + " nonnegative memory")
        if policy == "disabled":
            audit.equal(int(row["cache_bytes"]), 0, label + " no stored cache")
        if policy in {"disabled", "reset_each_request"} or index == 0:
            audit.equal(int(row["cache_bytes_before_compile"]), 0, label + " empty starting cache")
        audit.equal(row["phase"], "first_cold" if index == 0 else "after_first", label + " cold versus later timing")
        audit.equal(boolean(row["whole_grammar_reused"]), metadata["whole_grammar_reused"], label + " reuse flag")
        for overlap in ("tool_overlap_with_previous_requests", "parameter_structure_overlap_with_previous_requests"):
            audit.close(row[overlap], metadata[overlap], label + " raw overlap")
        pairs[(*key, index)].append((policy, int(row["pair_order_position"])))
        conditions[(row["mode"], count, policy, row["phase"])].append(row)
    audit.equal(set(pairs), {(*key, index) for key in expected_blocks for index in range(requests)}, "complete three-control coverage")
    for key, pair in pairs.items():
        number = (key[0] + key[3]) % 6
        offset = number % 3
        order = POLICIES[offset:] + POLICIES[:offset]
        if number >= 3:
            order.reverse()
        audit.equal(pair, list(zip(order, range(3))), "balanced six-order execution " + str(key))
    summaries = {(item["mode"], item["tool_count"], item["cache_policy"], item["phase"]): item
                 for item in summary["condition_summaries"]}
    audit.equal(len(summaries), len(summary["condition_summaries"]), "no duplicate condition summary groups")
    audit.equal(set(summaries), set(conditions), "all condition summaries present")
    for key, selected in conditions.items():
        values = [float(row["compile_ms"]) for row in selected]
        item = summaries[key]
        audit.equal(item["observations"], len(values), str(key) + " count")
        expected = {"mean_compile_ms": statistics.mean(values), "median_compile_ms": statistics.median(values),
                    "min_compile_ms": min(values), "max_compile_ms": max(values),
                    "mean_cache_bytes": statistics.mean(int(row["cache_bytes"]) for row in selected)}
        for field, value in expected.items():
            audit.close(item[field], value, str(key) + " independently recomputed " + field)
    rng = random.Random(summary["seed"] + 90210)
    for field, baseline in [("paired_after_first_speedups", "disabled"),
                            ("cross_request_after_first_speedups", "reset_each_request")]:
        groups = {(item["mode"], item["tool_count"]): item for item in summary[field]}
        audit.equal(len(groups), len(summary[field]), field + " no duplicate groups")
        audit.equal(set(groups), {(mode, count) for mode in ("static", "dynamic") for count in counts}, field + " complete groups")
        for mode, count in sorted(groups):
            item, ratios = groups[(mode, count)], []
            audit.equal(item["baseline_policy"], baseline, "distinct baseline control")
            audit.equal(item["target_policy"], "persistent", "persistent comparison target")
            for repeat in range(repeats):
                means = {policy: statistics.mean(float(row["compile_ms"]) for row in rows if row["mode"] == mode
                         and int(row["tool_count"]) == count and int(row["repeat"]) == repeat
                         and row["cache_policy"] == policy and row["phase"] == "after_first")
                         for policy in (baseline, "persistent")}
                ratio = means[baseline] / means["persistent"]
                ratios.append(ratio)
                detail = item["repeat_speedups"][repeat]
                audit.equal(detail["repeat"], repeat, "independent repeat identity")
                audit.close(detail["baseline_mean_ms"], means[baseline], "baseline mean")
                audit.close(detail["persistent_cache_mean_ms"], means["persistent"], "persistent mean")
                audit.close(detail["speedup"], ratio, "paired repeat ratio")
            audit.equal(item["repeat_count"], repeats, "independent repeat units")
            audit.close(item["mean_speedup"], statistics.mean(ratios), "mean of repeat ratios")
            bootstrap = sorted(statistics.mean(rng.choices(ratios, k=repeats)) for _ in range(2000))
            interval = item["bootstrap_95pct_interval"]
            audit.check(isinstance(interval, list) and len(interval) == 2, "full benchmark has two confidence interval endpoints")
            if isinstance(interval, list) and len(interval) == 2:
                for recorded, expected in zip(interval, [bootstrap[49], bootstrap[1949]]):
                    audit.close(recorded, expected, "seeded repeat-level bootstrap")
    required = {"valid_beijing", "valid_paris", "invalid_enum", "invalid_type", "invalid_missing_required",
                "invalid_extra_property", "cpu_kernel_independent_bitmask_unpack"}
    audit.equal({case["case"] for case in summary["mask_checks"]}, required, "all mask fixture cases")
    audit.equal(len(summary["mask_checks"]), len(required), "no repeated or substituted mask cases")
    audit.check(all(case["passed"] is True for case in summary["mask_checks"]), "saved mask correctness checks")
    audit.check(summary["all_invariants_passed"] is True and all(summary["invariants"].values()), "reported invariants")


def notebook(audit, path):
    import nbformat
    document = nbformat.read(path, as_version=4)
    nbformat.validate(document)
    cells = [cell for cell in document.cells if cell.cell_type == "code"]
    audit.check(bool(cells), "notebook has experiment code cells")
    counts = []
    for index, cell in enumerate(cells):
        count = cell.get("execution_count")
        audit.check(type(count) is int and count > 0, f"notebook cell {index + 1} executed")
        counts.append(count)
        audit.check(not any(output.output_type == "error" for output in cell.get("outputs", [])),
                    f"notebook cell {index + 1} has no error")
    audit.check(all(type(count) is int for count in counts) and counts == sorted(set(counts)),
                "notebook records one ordered executed run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--generation-dir", type=Path, default=Path("results"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("results/cache_benchmark"))
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--report", type=Path, default=Path("results/validation_report.json"))
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    audit = Audit()
    audit.stage("independent_checker_self_tests", lambda: self_tests(audit))
    if not args.self_test_only:
        audit.stage("pinned_provenance", lambda: provenance(audit, root))
        audit.stage("paired_generation", lambda: generation(audit, root, resolve(args.generation_dir)))
        audit.stage("cache_benchmark", lambda: benchmark(audit, root, resolve(args.benchmark_dir)))
        if args.notebook is not None:
            audit.stage("executed_notebook", lambda: notebook(audit, resolve(args.notebook)))
    report = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "checker self-tests only; NOT replication verification" if args.self_test_only else
                 "independent offline audit of fixed provenance, paired model generation, CPU cache benchmark and requested notebook",
        "experiment_runners_imported": False, "network_used": False, "self_test_only": args.self_test_only,
        "notebook_checked": args.notebook is not None and not args.self_test_only,
        "checks": sum(section["checks"] for section in audit.sections.values()),
        "passed_checks": sum(section["passed"] for section in audit.sections.values()),
        "all_passed": all(not section["failures"] for section in audit.sections.values()), "sections": audit.sections}
    path = resolve(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("scope", "checks", "passed_checks", "all_passed")}, indent=2))
    for name, section in report["sections"].items():
        if section["failures"]:
            print(name + ": " + "\n".join(section["failures"][:20]))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

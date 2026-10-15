#!/usr/bin/env python3
"""Local, single-threaded mechanism replication of XGrammar-2 section 4.1.

This is an independent CPU harness using the unmodified official xgrammar 0.2.0
library. It does not reproduce SGLang, GPU throughput, or the paper's absolute
timings. The dynamic workload never repeats a complete tool-set grammar; it can
reuse only substructures. Static requests intentionally repeat one grammar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TYPE_ALIASES = {
    "dict": "object", "list": "array", "tuple": "array", "float": "number",
    "int": "integer", "str": "string", "bool": "boolean", "None": "null",
    "NoneType": "null", "any": "string", "Any": "string",
}
JSON_TYPES = {"object", "array", "number", "integer", "string", "boolean", "null"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize_schema(value, changes, path="parameters"):
    """Translate documented BFCL type aliases, leaving constraints intact.

    Descriptions, required properties, ranges, enums, arrays, and nesting are
    retained. Unknown type names fail explicitly rather than becoming invented
    schemas. BFCL's Python schema mapping treats 'any' as string; this explicit
    compatibility translation is counted rather than guessing a new type.
    """
    if isinstance(value, list):
        return [normalize_schema(x, changes, f"{path}[{i}]") for i, x in enumerate(value)]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "type":
            types = item if isinstance(item, list) else [item]
            mapped = []
            for name in types:
                replacement = TYPE_ALIASES.get(name, name)
                if replacement not in JSON_TYPES:
                    raise ValueError(f"Unknown BFCL type {name!r} at {path}")
                mapped.append(replacement)
                if replacement != name:
                    changes[f"{name} -> {replacement}"] += 1
            if mapped:
                result[key] = mapped if isinstance(item, list) else mapped[0]
        elif key in {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}:
            result[key] = {name: normalize_schema(child, changes, f"{path}.{key}.{name}")
                           for name, child in item.items()}
        elif key in {"items", "additionalProperties", "contains", "propertyNames", "if", "then",
                     "else", "not", "unevaluatedProperties", "unevaluatedItems", "additionalItems",
                     "allOf", "anyOf", "oneOf", "prefixItems"}:
            result[key] = normalize_schema(item, changes, f"{path}.{key}")
        else:
            # Enum/default/example payloads and property names are data, not schemas.
            result[key] = item
    return result


def load_tools(path):
    import jsonschema

    raw_bytes = path.read_bytes()
    records = [json.loads(line) for line in raw_bytes.decode().splitlines() if line.strip()]
    tools_by_identity = {}
    changes = Counter()
    skipped = []
    duplicates = 0
    for record in records:
        functions = record.get("function", record.get("tools", []))
        if isinstance(functions, dict):
            functions = [functions]
        for function in functions:
            if function.get("type") == "function" and "function" in function:
                function = function["function"]
            name = function.get("name")
            if not isinstance(name, str) or not name or "<" in name or ">" in name:
                skipped.append({"id": record.get("id"), "reason": "invalid tool name"})
                continue
            try:
                per_tool_changes = Counter()
                parameters = normalize_schema(function.get("parameters", {}), per_tool_changes)
                jsonschema.Draft202012Validator.check_schema(parameters)
            except (ValueError, jsonschema.SchemaError) as exc:
                skipped.append({"id": record.get("id"), "name": name, "reason": str(exc)[:500]})
                continue
            tool = {"name": name, "parameters": parameters}
            identity = digest(tool)
            if identity in tools_by_identity:
                duplicates += 1
                continue
            tools_by_identity[identity] = tool
            changes.update(per_tool_changes)
    tools = sorted(tools_by_identity.values(), key=canonical)
    return tools, {
        "input_file": path.name,
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "input_record_count": len(records),
        "distinct_tool_count": len(tools),
        "distinct_tool_name_count": len({x["name"] for x in tools}),
        "exact_duplicate_tool_count": duplicates,
        "same_name_schema_variants": len(tools) - len({x["name"] for x in tools}),
        "normalization_counts": dict(changes),
        "skipped_records": skipped,
    }


def structural_tag(tools):
    """Use the official TagDispatch-backed triggered_tags representation."""
    return {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {"type": "tag", "begin": f"<function={tool['name']}>",
                 "content": {"type": "json_schema", "json_schema": tool["parameters"]},
                 "end": "</function>"}
                for tool in sorted(tools, key=lambda x: x["name"])
            ],
        },
    }


def make_workloads(tools, count, requests, seed):
    rng = random.Random(seed)
    by_name = defaultdict(list)
    for index, tool in enumerate(tools):
        by_name[tool["name"]].append(index)
    names = sorted(by_name)

    def sample_unique_names():
        return tuple(sorted(rng.choice(by_name[name]) for name in rng.sample(names, count)))

    first = sample_unique_names()
    dynamic_indices = [first]
    seen = {first}
    while len(dynamic_indices) < requests:
        candidate = sample_unique_names()
        if candidate not in seen:
            dynamic_indices.append(candidate)
            seen.add(candidate)
    return {"static": [first] * requests, "dynamic": dynamic_indices}


def workload_metadata(tools, indices):
    previous_names, previous_parameter_hashes, previous_grammars = set(), set(), set()
    metadata = []
    for request_index, selection in enumerate(indices):
        subset = [tools[i] for i in selection]
        grammar = structural_tag(subset)
        names = {x["name"] for x in subset}
        parameter_hashes = {digest(x["parameters"]) for x in subset}
        grammar_hash = digest(grammar)
        metadata.append({
            "request_index": request_index,
            "grammar_sha256": grammar_hash,
            "tool_names": sorted(names),
            "selected_tool_sha256": [digest(tool) for tool in sorted(subset, key=lambda x: x["name"])],
            "whole_grammar_reused": grammar_hash in previous_grammars,
            "tool_overlap_with_previous_requests": len(names & previous_names) / len(names),
            "parameter_structure_overlap_with_previous_requests":
                len(parameter_hashes & previous_parameter_hashes) / len(parameter_hashes),
            "grammar_json_bytes": len(canonical(grammar).encode()),
        })
        previous_names.update(names)
        previous_parameter_hashes.update(parameter_hashes)
        previous_grammars.add(grammar_hash)
    return metadata


def token_allowed(mask, token_id):
    word = int(mask[0, token_id // 32].item()) & 0xFFFFFFFF
    return bool((word >> (token_id % 32)) & 1)


def check_mask_sequences(xgr, torch, tokenizer, tokenizer_info):
    """Independent fixture tests, separate from the public benchmark workload."""
    import jsonschema

    weather_schema = {
        "type": "object", "properties": {
            "city": {"type": "string", "enum": ["Beijing", "Paris"]},
            "units": {"type": "string", "enum": ["C", "F"]},
        }, "required": ["city", "units"], "additionalProperties": False,
    }
    tools = [{"name": "get_weather", "parameters": weather_schema}]
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1, cache_enabled=False)
    compiled = compiler.compile_structural_tag(structural_tag(tools))
    cases = [
        ("valid_beijing", {"city": "Beijing", "units": "C"}, True),
        ("valid_paris", {"city": "Paris", "units": "F"}, True),
        ("invalid_enum", {"city": "London", "units": "C"}, False),
        ("invalid_type", {"city": 1, "units": "C"}, False),
        ("invalid_missing_required", {"city": "Beijing"}, False),
        ("invalid_extra_property", {"city": "Beijing", "units": "C", "extra": 1}, False),
    ]
    rows = []
    vocab_size = len(tokenizer)
    for name, arguments, expected in cases:
        text = "Calling tool. <function=get_weather>" + json.dumps(
            arguments, separators=(",", ":"), ensure_ascii=False
        ) + "</function>"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        matcher = xgr.GrammarMatcher(compiled)
        bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        rejected_at = None
        mask_agreement = True
        per_token_us = []
        for index, token_id in enumerate(token_ids):
            start = time.perf_counter_ns()
            needs_apply = matcher.fill_next_token_bitmask(bitmask)
            per_token_us.append((time.perf_counter_ns() - start) / 1000)
            allowed = not needs_apply or token_allowed(bitmask, token_id)
            accepted = matcher.accept_token(token_id)
            mask_agreement = mask_agreement and allowed == accepted
            if not accepted:
                rejected_at = index
                break
        sequence_accepted = rejected_at is None
        independently_valid = jsonschema.Draft202012Validator(weather_schema).is_valid(arguments)
        complete = False
        if sequence_accepted:
            eos = tokenizer.eos_token_id
            matcher.fill_next_token_bitmask(bitmask)
            complete = token_allowed(bitmask, eos) and matcher.accept_token(eos)
        rows.append({
            "case": name, "expected_valid": expected,
            "independent_jsonschema_valid": independently_valid,
            "sequence_accepted": sequence_accepted,
            "complete_stop_token_accepted": complete,
            "mask_accept_agreement": mask_agreement,
            "rejected_at_token_index": rejected_at,
            "token_count": len(token_ids),
            "mask_generation_mean_us": statistics.mean(per_token_us),
            "passed": independently_valid == expected and sequence_accepted == expected
                      and mask_agreement and (not expected or complete),
        })
    # Independently unpack the bitmask to validate the CPU masking kernel.
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_string("Calling tool. <function=get_weather>")
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    logits = torch.arange(vocab_size, dtype=torch.float32).reshape(1, -1)
    original = logits.clone()
    xgr.apply_token_bitmask_inplace(logits, bitmask, backend="cpu")
    expected_allowed = torch.tensor([token_allowed(bitmask, i) for i in range(vocab_size)])
    kernel_agrees = bool(torch.equal(torch.isfinite(logits[0]), expected_allowed))
    retained_logits = bool(torch.equal(logits[0, expected_allowed], original[0, expected_allowed]))
    rows.append({"case": "cpu_kernel_independent_bitmask_unpack", "passed":
                 kernel_agrees and retained_logits,
                 "finite_logits_match_independent_bits": kernel_agrees,
                 "allowed_logits_unchanged": retained_logits})
    return rows


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, repeat_count, seed):
    groups = defaultdict(list)
    for row in rows:
        if row["error"]:
            continue
        groups[(row["mode"], row["tool_count"], row["cache_policy"], row["phase"])].append(row)
    summaries = []
    for (mode, count, policy, phase), group in sorted(groups.items()):
        values = [x["compile_ms"] for x in group]
        summaries.append({"mode": mode, "tool_count": count, "cache_policy": policy,
                          "cache_enabled": policy != "disabled",
                          "phase": phase, "observations": len(values),
                          "mean_compile_ms": statistics.mean(values),
                          "median_compile_ms": statistics.median(values),
                          "min_compile_ms": min(values), "max_compile_ms": max(values),
                          "mean_cache_bytes": statistics.mean(x["cache_bytes"] for x in group)})
    paired, cross_request_paired = [], []
    keys = sorted({(r["mode"], r["tool_count"]) for r in rows})
    bootstrap_rng = random.Random(seed + 90210)
    for baseline_policy, output_list in [("disabled", paired),
                                         ("reset_each_request", cross_request_paired)]:
        for mode, count in keys:
            ratios = []
            repeat_details = []
            for repeat in range(repeat_count):
                values = {}
                for policy in [baseline_policy, "persistent"]:
                    subset = [r["compile_ms"] for r in rows if r["mode"] == mode
                              and r["tool_count"] == count and r["repeat"] == repeat
                              and r["cache_policy"] == policy and r["phase"] == "after_first"
                              and not r["error"]]
                    if subset:
                        values[policy] = statistics.mean(subset)
                if len(values) == 2:
                    ratio = values[baseline_policy] / values["persistent"]
                    ratios.append(ratio)
                    repeat_details.append({"repeat": repeat,
                                           "baseline_mean_ms": values[baseline_policy],
                                           "persistent_cache_mean_ms": values["persistent"],
                                           "speedup": ratio})
            if ratios:
                boot = sorted(statistics.mean(bootstrap_rng.choices(ratios, k=len(ratios)))
                              for _ in range(2000))
                output_list.append({"mode": mode, "tool_count": count,
                                    "baseline_policy": baseline_policy, "target_policy": "persistent",
                                    "repeat_speedups": repeat_details,
                                    "mean_speedup": statistics.mean(ratios),
                                    "bootstrap_95pct_interval": [boot[49], boot[1949]] if len(ratios) >= 3 else None,
                                    "interval_unit": "independent workload repeat",
                                    "interval_estimation_note": "Reported only for at least three repeats; "
                                                                "few-repeat intervals should be interpreted cautiously.",
                                    "repeat_count": len(ratios)})
    return summaries, paired, cross_request_paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "inputs/bfcl_simple.jsonl")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                        help="Pinned local model/tokenizer directory is preferred.")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--counts", default="10,50,100,200")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/cache_benchmark")
    parser.add_argument("--smoke", action="store_true",
                        help="3 requests, 1 repeat, counts 3/10; never labelled full verification.")
    parser.add_argument("--skip-mask-checks", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.requests, args.repeats, args.counts = 3, 1, "3,10"
    if args.requests < 2 or args.repeats < 1:
        parser.error("at least 2 requests and 1 repeat are required")
    if not args.smoke and args.repeats < 3:
        parser.error("a complete benchmark requires at least 3 independent repeats")
    requested_counts = sorted({int(x) for x in args.counts.split(",")})
    if not requested_counts or min(requested_counts) < 1:
        parser.error("counts must contain positive integers")

    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer

    version = importlib.metadata.version("xgrammar")
    if version != "0.2.0":
        raise RuntimeError(f"This pinned experiment requires xgrammar==0.2.0; found {version}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    tools, input_metadata = load_tools(args.input)
    distinct_names = len({tool["name"] for tool in tools})
    counts = [n for n in requested_counts if n < distinct_names
              and math.comb(distinct_names, n) >= args.requests]
    if not counts:
        raise ValueError("Tool pool is too small to create unique dynamic requests at these counts")
    omitted = [n for n in requested_counts if n not in counts]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision,
        local_files_only=args.local_files_only, trust_remote_code=False,
    )
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"XGrammar {version}; CPU max_threads=1; {len(tools)} public tools; counts={counts}", flush=True)
    rows, workload_manifest = [], []
    started = time.perf_counter()
    for repeat in range(args.repeats):
        repeat_seed = args.seed + repeat * 100003
        for count in counts:
            workloads = make_workloads(tools, count, args.requests, repeat_seed + count)
            for mode, indices in workloads.items():
                metadata = workload_metadata(tools, indices)
                workload_manifest.append({"repeat": repeat, "seed": repeat_seed + count,
                                          "mode": mode, "tool_count": count, "requests": metadata})
                if mode == "dynamic":
                    assert len({x["grammar_sha256"] for x in metadata}) == args.requests
                    assert not any(x["whole_grammar_reused"] for x in metadata)
                else:
                    assert len({x["grammar_sha256"] for x in metadata}) == 1
                policies = ["disabled", "persistent", "reset_each_request"]
                compilers = {
                    policy: xgr.GrammarCompiler(tokenizer_info, max_threads=1,
                                               cache_enabled=policy != "disabled")
                    for policy in policies
                }
                for request_index, selection in enumerate(indices):
                    grammar = structural_tag([tools[i] for i in selection])
                    # Same schema in all controls; order cycles through six permutations.
                    order_number = (repeat + request_index) % 6
                    offset = order_number % 3
                    order = policies[offset:] + policies[:offset]
                    if order_number >= 3:
                        order.reverse()
                    for order_position, policy in enumerate(order):
                        compiler = compilers[policy]
                        if policy == "reset_each_request":
                            # Retain within-grammar caching, eliminate all cross-request reuse.
                            # A fresh compiler also avoids version 0.2.0's residual
                            # cache-size accounting after clear_cache(). Construction
                            # and destruction are excluded from compile timing.
                            compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1,
                                                           cache_enabled=True)
                            compilers[policy] = compiler
                        cache_before = compiler.get_cache_size_bytes()
                        error = ""
                        start = time.perf_counter_ns()
                        try:
                            compiled = compiler.compile_structural_tag(grammar)
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            compiled = None
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        row = {
                            "repeat": repeat, "seed": repeat_seed + count,
                            "mode": mode, "tool_count": count, "request_index": request_index,
                            "phase": "first_cold" if request_index == 0 else "after_first",
                            "cache_policy": policy, "cache_enabled": policy != "disabled",
                            "pair_order_position": order_position,
                            "compile_ms": elapsed, "cache_bytes": compiler.get_cache_size_bytes(),
                            "cache_bytes_before_compile": cache_before,
                            "compiled_grammar_bytes": compiled.memory_size_bytes if compiled else 0,
                            "grammar_sha256": metadata[request_index]["grammar_sha256"],
                            "whole_grammar_reused": metadata[request_index]["whole_grammar_reused"],
                            "tool_overlap_with_previous_requests":
                                metadata[request_index]["tool_overlap_with_previous_requests"],
                            "parameter_structure_overlap_with_previous_requests":
                                metadata[request_index]["parameter_structure_overlap_with_previous_requests"],
                            "error": error,
                        }
                        rows.append(row)
                        del compiled
                    timings = {r["cache_policy"]: r["compile_ms"] for r in rows[-3:]}
                    print(f"repeat {repeat + 1}/{args.repeats} {mode} tools={count} "
                          f"request {request_index + 1}/{args.requests}: "
                          f"off={timings['disabled']:.2f} ms on={timings['persistent']:.2f} ms "
                          f"reset={timings['reset_each_request']:.2f} ms", flush=True)
                # Persist completed blocks; a later interruption does not erase evidence.
                write_csv(args.output_dir / "raw.csv", rows)
                del compilers

    mask_checks = [] if args.skip_mask_checks else check_mask_sequences(
        xgr, torch, tokenizer, tokenizer_info
    )
    summaries, paired, cross_request_paired = summarize(rows, args.repeats, args.seed)
    dynamic_rows = [r for r in rows if r["mode"] == "dynamic"]
    invariants = {
        "dynamic_complete_grammars_never_repeated": not any(r["whole_grammar_reused"] for r in dynamic_rows),
        "all_compilations_succeeded": not any(r["error"] for r in rows),
        "cache_off_memory_is_zero": all(r["cache_bytes"] == 0 for r in rows if not r["cache_enabled"]),
        "reset_control_starts_every_request_with_empty_cache": all(
            r["cache_bytes_before_compile"] == 0 for r in rows
            if r["cache_policy"] == "reset_each_request"
        ),
        "paired_identical_input_hashes": all(
            len({r["grammar_sha256"] for r in rows if r["repeat"] == repeat
                 and r["mode"] == mode and r["tool_count"] == count
                 and r["request_index"] == index}) == 1
            for repeat in range(args.repeats) for count in counts
            for mode in ["static", "dynamic"] for index in range(args.requests)
        ),
        "all_expected_observations_present": len(rows) == args.repeats * len(counts) * args.requests * 6,
        "single_thread_configured": torch.get_num_threads() == 1,
        "independent_mask_checks_passed": bool(mask_checks) and all(x["passed"] for x in mask_checks),
    }
    if args.skip_mask_checks:
        invariants.pop("independent_mask_checks_passed")
    summary = {
        "scope": "XGrammar-2 section 4.1 single-threaded CPU mechanism replication; "
                 "independent harness, official library, public BFCL schemas. "
                 "Not a complete SGLang/GPU throughput or paper-timing replication.",
        "paper": "https://arxiv.org/abs/2601.04426",
        "xgrammar_version": version,
        "author_source_commit": "5433fdbea7003eea61cb863768a05fb8da3bf919",
        "smoke_run": args.smoke,
        "complete_repeat_protocol": not args.smoke and args.repeats >= 3,
        "seed": args.seed, "repeats": args.repeats, "requests_per_condition": args.requests,
        "requested_tool_counts": requested_counts, "executed_tool_counts": counts,
        "omitted_tool_counts_due_to_pool_size_or_unique_combination_limit": omitted,
        "tokenizer": Path(args.tokenizer).name if Path(args.tokenizer).exists() else args.tokenizer,
        "tokenizer_revision_argument": args.tokenizer_revision,
        "python": platform.python_version(), "architecture": platform.machine(),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "max_compiler_threads": 1, "torch_threads": torch.get_num_threads(),
        "timing_definition": "perf_counter_ns around compile_structural_tag only; "
                             "compiler construction, JSON preparation and cache-size inspection excluded",
        "cache_state_definition": "Fresh separate compiler per repeat/mode/count and cache condition. "
                                  "Request 0 is cold; after_first means subsequent requests. "
                                  "Disabled compiler remains uncached for every request. "
                                  "reset_each_request has a new cache_enabled=True compiler before every request; "
                                  "it retains within-grammar reuse while isolating cross-request reuse. "
                                  "Static cache reuses whole grammars; dynamic cache can only reuse substructures.",
        "pairing": "Identical workloads across disabled, persistent and reset_each_request controls; "
                   "execution order cycles through six permutations. "
                   "Independent seeded workloads per repeat; 95% bootstrap interval resamples repeat means.",
        "sampling": "Uniformly sample distinct public BFCL tool names, then uniformly select an original "
                    "schema variant for each sampled name. Different schemas with the same tool name "
                    "are retained in the pool but never collide inside a request. No tool names are invented.",
        "input": input_metadata,
        "condition_summaries": summaries,
        "paired_after_first_speedups": paired,
        "cross_request_after_first_speedups": cross_request_paired,
        "comparison_definition": "disabled/persistent measures all cache benefits, including within-grammar reuse. "
                                 "reset_each_request/persistent isolates cross-request cache benefits. "
                                 "For dynamic requests, complete grammar hashes never repeat, so these are substructure benefits.",
        "invariants": invariants,
        "mask_checks": mask_checks,
        "wall_seconds": time.perf_counter() - started,
        "all_invariants_passed": all(invariants.values()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "workload_manifest.json").write_text(json.dumps(workload_manifest, indent=2) + "\n")
    write_csv(args.output_dir / "condition_summary.csv", summaries)
    if mask_checks:
        write_csv(args.output_dir / "mask_checks.csv", mask_checks)
    print(json.dumps({"all_invariants_passed": summary["all_invariants_passed"],
                      "observations": len(rows), "wall_seconds": summary["wall_seconds"],
                      "smoke_run": args.smoke}), flush=True)
    if not summary["all_invariants_passed"]:
        raise SystemExit("One or more invariants failed; inspect summary.json. Performance ratios are observations, not correctness gates.")


if __name__ == "__main__":
    main()

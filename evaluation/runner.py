"""Quantitative evaluation harness for logseq-chat configurations.

Running ``python evaluation/runner.py`` (or ``make evaluate``) executes the
following steps:

1. Load the user's base configuration from ``config.yaml``.
2. Expand it with the presets listed in ``evaluations/configurations.yaml``.
3. Re-ingest the Logseq graph for each preset into an isolated storage
   directory.
4. Query the index with prompts from ``evaluations/datasets/baseline.yaml``.
5. Compute accuracy, coverage, hallucination, relevance, and latency metrics.
6. Report a ranked leaderboard and identify the best configuration.

The script is intentionally deterministic and free of network calls so that
users can iterate rapidly on new presets or datasets.  Replace the sample
queries with your own gold-standard answers to turn this into a bespoke tuning
loop for your notes.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from app_config import ConfigNamespace, apply_overrides, load_app_config
from chat import build_query_engine
from ingest import run_ingest
from query_expansion import expand_query


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of {path}, got {type(data)!r}")
    return data


def _ensure_profiled_storage(config: ConfigNamespace, profile: str) -> ConfigNamespace:
    updated = config.to_dict()
    base = Path(updated["storage"]["chroma_path"])
    profile_root = Path(".rag") / "evaluations" / profile
    updated["storage"]["chroma_path"] = str(profile_root / "chroma")
    updated["storage"]["collection_name"] = f"{updated['storage'].get('collection_name', 'logseq_rag')}_{profile}"
    updated["storage"]["clear_before_ingest"] = True
    return ConfigNamespace(updated)


def _extract_sources(response: Any) -> List[str]:
    sources: List[str] = []
    for node in getattr(response, "source_nodes", []) or []:
        metadata = getattr(getattr(node, "node", node), "metadata", None) or {}
        source = metadata.get("source") or metadata.get("file_path")
        if source:
            sources.append(str(source))
    return sources


def _lower_list(values: Iterable[str]) -> List[str]:
    return [v.lower() for v in values]


def _score_query(response_text: str, sources: List[str], spec: Dict[str, Any]) -> Dict[str, float]:
    text = response_text.lower()
    keywords = spec.get("answer_keywords", {}) or {}
    required = _lower_list(keywords.get("required", []) or [])
    optional = _lower_list(keywords.get("optional", []) or [])

    required_hits = sum(1 for kw in required if kw in text)
    optional_hits = sum(1 for kw in optional if kw in text)

    accuracy = required_hits / len(required) if required else 1.0
    relevance = (required_hits + optional_hits) / (len(required) + len(optional)) if (required or optional) else 1.0

    expected_sources = _lower_list(spec.get("expected_sources", []) or [])
    sources_lower = _lower_list(sources)
    matched_sources = sum(1 for src in sources_lower if src in expected_sources)
    coverage = matched_sources / len(expected_sources) if expected_sources else 1.0

    allow_extra = bool(spec.get("allow_additional_sources", False))
    hallucinations = 0
    if not allow_extra and sources_lower:
        hallucinations = sum(1 for src in sources_lower if src not in expected_sources)
    hallucination_rate = hallucinations / len(sources_lower) if sources_lower else 0.0

    return {
        "accuracy": accuracy,
        "relevance": relevance,
        "coverage": coverage,
        "hallucination_rate": hallucination_rate,
    }


def evaluate_configuration(
    name: str,
    base_config: ConfigNamespace,
    overrides: Dict[str, Any],
    dataset: Dict[str, Any],
    *,
    max_queries: int | None,
) -> Tuple[ConfigNamespace, Dict[str, Any]]:
    config = apply_overrides(base_config, overrides)
    profiled = _ensure_profiled_storage(config, name)

    run_ingest(profiled, verbose=False)
    query_engine = build_query_engine(profiled)

    queries = dataset.get("queries", [])
    results: List[Dict[str, float]] = []

    for spec in queries[: max_queries or len(queries)]:
        expanded = expand_query(spec["question"], profiled)
        start = time.perf_counter()
        response = query_engine.query(expanded.expanded)
        latency_ms = (time.perf_counter() - start) * 1000

        text = getattr(response, "response", "")
        sources = _extract_sources(response)
        metrics = _score_query(text, sources, spec)
        metrics["latency_ms"] = latency_ms
        results.append(metrics)

    aggregate: Dict[str, float] = {}
    if results:
        for key in ("accuracy", "relevance", "coverage", "hallucination_rate"):
            aggregate[key] = statistics.mean(r[key] for r in results)
        aggregate["avg_latency_ms"] = statistics.mean(r["latency_ms"] for r in results)
    else:
        aggregate = {"accuracy": 0.0, "relevance": 0.0, "coverage": 0.0, "hallucination_rate": 1.0, "avg_latency_ms": float("inf")}

    return profiled, {"metrics": aggregate, "raw": results}


def _composite_score(metrics: Dict[str, float], weights: Dict[str, float], speed_anchor: float) -> float:
    latency = metrics.get("avg_latency_ms", float("inf"))
    speed_score = 0.0
    if speed_anchor > 0 and latency > 0:
        speed_score = min(speed_anchor / latency, 1.0)

    return (
        weights.get("accuracy_weight", 0.0) * metrics.get("accuracy", 0.0)
        + weights.get("coverage_weight", 0.0) * metrics.get("coverage", 0.0)
        + weights.get("relevance_weight", 0.0) * metrics.get("relevance", 0.0)
        + weights.get("hallucination_weight", 0.0) * (1 - metrics.get("hallucination_rate", 0.0))
        + weights.get("speed_weight", 0.0) * speed_score
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate configuration presets against a labelled dataset.")
    parser.add_argument("--dataset", type=Path, default=None, help="Override dataset path")
    parser.add_argument("--configurations", type=Path, default=None, help="Override configurations list")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit the number of queries executed per configuration")
    args = parser.parse_args()

    base_config = load_app_config()
    evaluation_cfg = base_config.evaluation

    dataset_path = args.dataset or Path(evaluation_cfg.dataset)
    config_list_path = args.configurations or Path(evaluation_cfg.configurations_file)
    max_queries = args.max_queries if args.max_queries is not None else evaluation_cfg.max_queries

    dataset = _load_yaml(dataset_path)
    config_specs = _load_yaml(config_list_path).get("configurations", [])

    if not config_specs:
        raise SystemExit(f"No configurations listed in {config_list_path}")

    weights = evaluation_cfg.scoring.to_dict() if hasattr(evaluation_cfg.scoring, "to_dict") else dict(evaluation_cfg.scoring)

    leaderboard: List[Tuple[str, float, Dict[str, Any]]] = []
    best_config: Tuple[str, ConfigNamespace, Dict[str, Any]] | None = None
    latencies: List[float] = []
    per_config_results: Dict[str, Dict[str, Any]] = {}

    for entry in config_specs:
        name = entry.get("name")
        if not name:
            raise ValueError(f"Configuration entry missing name: {entry}")
        preset_path = entry.get("preset")
        overrides: Dict[str, Any] = {}
        if preset_path:
            overrides = _load_yaml(Path(preset_path))
        profile_config, result = evaluate_configuration(
            name,
            base_config,
            overrides,
            dataset,
            max_queries=max_queries,
        )
        metrics = result["metrics"]
        latencies.append(metrics.get("avg_latency_ms", 0.0))
        per_config_results[name] = {"config": profile_config.to_dict(), **result}
        leaderboard.append((name, 0.0, metrics))

    if not leaderboard:
        raise SystemExit("No evaluation results produced.")

    speed_anchor = min((metrics.get("avg_latency_ms", float("inf")) for _, _, metrics in leaderboard), default=0.0)

    adjusted: List[Tuple[str, float, Dict[str, Any]]] = []
    for name, _, metrics in leaderboard:
        score = _composite_score(metrics, weights, speed_anchor)
        adjusted.append((name, score, metrics))
        if best_config is None or score > best_config[2]["metrics"]["score"]:
            best_config = (name, per_config_results[name]["config"], {"metrics": {**metrics, "score": score}})

    adjusted.sort(key=lambda item: item[1], reverse=True)

    print("\nConfiguration leaderboard:")
    print("-------------------------")
    for name, score, metrics in adjusted:
        print(
            f"{name:>15}  score={score:0.3f}  accuracy={metrics['accuracy']:0.3f}  "
            f"coverage={metrics['coverage']:0.3f}  relevance={metrics['relevance']:0.3f}  "
            f"hallucinations={metrics['hallucination_rate']:0.3f}  avg_latency_ms={metrics['avg_latency_ms']:0.1f}"
        )

    if best_config:
        best_name, best_settings, payload = best_config
        print(
            "\nBest configuration:",
            best_name,
            f"(score={payload['metrics']['score']:0.3f})",
        )
        output_path = Path("evaluations/results")
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "latest.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {
                    "best_configuration": best_name,
                    "score": payload["metrics"]["score"],
                    "metrics": {k: v for k, v in payload["metrics"].items() if k != "score"},
                },
                fh,
                sort_keys=False,
            )
        print(f"Saved summary to {output_path / 'latest.yaml'}")


if __name__ == "__main__":
    main()

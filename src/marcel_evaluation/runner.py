import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from pydantic import BaseModel

from marcel_evaluation.base import Metric, MultiGenerationMetricWrapper
from marcel_evaluation.metrics import (
    ROUGE,
    AnswerFaithfulness,
    AnswerRelevance,
    AnswerSimilarity,
    BERTScore,
    ContextLength,
    ContextPrecision,
    ContextRecall,
    CorpusBLEU,
    GeneratedAnswerLength,
    MeanReciprocalRank,
    NonAnswerCritic,
    PrecisionAtCutoff,
    RecallAtCutoff,
    ReferenceAnswerLength,
)
from marcel_evaluation.utils import load_run


class PydanticEncoder(json.JSONEncoder):
    """Some metrics return pydantic objects in their outputs (e.g., Verdicts). This custom encoder converts pydantic objects to plain python dict before json serialization."""

    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()  # Convert Pydantic model to dict
        if isinstance(obj, np.generic):
            return obj.item()  # convert numpy objects to plain python types
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # convert numpy objects to plain python types
        return super().default(obj)  # Handle other objects normally


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a run with specified model and metrics."
    )
    parser.add_argument(
        "--run_path", type=str, required=True, help="Path of the run to evaluate."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Huggingface/vLLM model identifier.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated list of metrics to run, or 'all' to run everything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override existing evaluation results if they exist.",
    )
    return parser.parse_args()


async def main(args):
    run_path = Path(args.run_path)
    output_json = run_path / "output.json"
    metrics_json = run_path / "metrics.json"
    metrics_raw_json = run_path / "metrics.raw.json"
    metrics_by_sample_json = run_path / "metrics_by_sample.json"
    config_json = run_path / "config.json"

    # Load existing raw metrics if they exist
    if metrics_json.exists() and not args.force:
        with open(metrics_raw_json) as f:
            results = json.load(f)
        with open(metrics_by_sample_json) as f:
            df_responses = pd.read_json(f)
    else:
        results = {}
        df_responses = None

    computed_metrics = set(results.keys())
    requested_metrics = (
        set(m.strip() for m in args.metrics.split(","))
        if not args.metrics.lower() == "all"
        else {
            "ROUGE",
            "AnswerFaithfulness",
            "AnswerRelevance",
            "AnswerSimilarity",
            "BERTScore",
            "ContextLength",
            "ContextPrecision",
            "ContextRecall",
            "CorpusBLEU",
            "GeneratedAnswerLength",
            "MeanReciprocalRank",
            "NonAnswerCritic",
            "PrecisionAtCutoff",
            "RecallAtCutoff",
            "ReferenceAnswerLength",
        }
    )
    computed_metrics = set(results.keys()) if not args.force else set()

    if "p@1" in computed_metrics:
        computed_metrics.add("PrecisionAtCutoff")
    if "r@1" in computed_metrics:
        computed_metrics.add("RecallAtCutoff")
    if "rouge1" in computed_metrics:
        computed_metrics.add("ROUGE")

    metrics = requested_metrics.difference(computed_metrics)

    if len(metrics) == 0:
        print(
            f"All requested metrics already computed in {metrics_raw_json}. Skipping."
        )
        return

    if config_json.exists():
        with open(config_json) as fin:
            config = json.load(fin)
    else:
        config = {}

    wandb_run = get_wandb_run(run_path, config)

    responses = load_run(output_json)
    if df_responses is None:
        df_responses = pd.DataFrame(responses)
    print(f"Evaluate: {output_json} (n = {len(responses)})")
    print(f"LLM-as-a-Judge: {args.model}")
    print(f"Metrics: {metrics}")

    selected_metrics: list[Metric] = []

    if "GeneratedAnswerLength" in metrics:
        selected_metrics.append(GeneratedAnswerLength())
    if "ReferenceAnswerLength" in metrics:
        selected_metrics.append(ReferenceAnswerLength())
    if "ContextLength" in metrics:
        selected_metrics.append(ContextLength())
    if "MeanReciprocalRank" in metrics:
        selected_metrics.append(MeanReciprocalRank())
    if "PrecisionAtCutoff" in metrics:
        for cutoff in [1, 3, 5, 10, 20, 30, 50]:
            selected_metrics.append(PrecisionAtCutoff(cutoff=cutoff))
    if "RecallAtCutoff" in metrics:
        for cutoff in [1, 3, 5, 10, 20, 30, 50]:
            selected_metrics.append(RecallAtCutoff(cutoff=cutoff))
    if "CorpusBLEU" in metrics:
        selected_metrics.append(CorpusBLEU())
    if "ROUGE" in metrics:
        selected_metrics.append(ROUGE(rouge_type="rouge1"))
        selected_metrics.append(ROUGE(rouge_type="rouge2"))
        selected_metrics.append(ROUGE(rouge_type="rougeLsum"))
    if "BERTScore" in metrics:
        selected_metrics.append(BERTScore(lang="en", rescale_with_baseline=True))
    if "AnswerSimilarity" in metrics:
        selected_metrics.append(AnswerSimilarity("all-mpnet-base-v2"))
    if "AnswerFaithfulness" in metrics:
        selected_metrics.append(AnswerFaithfulness(model=args.model))
    if "ContextPrecision" in metrics:
        selected_metrics.append(ContextPrecision(model=args.model))
    if "ContextRecall" in metrics:
        selected_metrics.append(ContextRecall(model=args.model))
    if "AnswerRelevance" in metrics:
        selected_metrics.append(
            AnswerRelevance(
                model=args.model, embedding_model_or_name="all-mpnet-base-v2"
            )
        )
    if "NonAnswerCritic" in metrics:
        selected_metrics.append(NonAnswerCritic(model=args.model))

    has_multiple_generated_answers = len(responses[0]["generated_answer"]) > 1
    if has_multiple_generated_answers:
        selected_metrics = [
            MultiGenerationMetricWrapper(metric)
            if metric.uses_generated_answer
            else metric
            for metric in selected_metrics
        ]

    # calculate metrics
    for i, metric in enumerate(selected_metrics):
        print(f"({i + 1}/{len(selected_metrics)}) Calculate {metric.name}")
        try:
            metric_data = metric.compute(responses)
        except NotImplementedError:
            metric_data = await metric.compute_async(responses)
        results[metric.name] = metric_data

        print(f"({i + 1}/{len(selected_metrics)}) Save {metric.name}")
        wandb_run.log({metric.name: metric_data["score"]})

        # report dataset-level scores
        aggregated_scores = {metric: data["score"] for metric, data in results.items()}
        with open(metrics_json, "w") as fout:
            json.dump(aggregated_scores, fout, indent=4)

        # report sample-level scores
        if metric_data.get("raw", []):
            df_responses[metric.name] = metric_data["raw"]

        df_responses.to_json(metrics_by_sample_json, orient="records")

        # persist raw metric results
        with open(metrics_raw_json, "w") as fout:
            json.dump(results, fout, cls=PydanticEncoder, indent=4)

    wandb_run.log({"responses": wandb.Table(dataframe=df_responses)})


def get_wandb_run(run_path, config):
    """Creates a new wandb run and persists its ID or resumes an existing one."""
    wandb_json = run_path / "wandb.json"
    if wandb_json.exists():
        with open(wandb_json) as fin:
            wandb_run_id = json.load(fin)["wandb_id"]
    else:
        wandb_run_id = None

    wandb_run = wandb.init(
        entity="mcmi",
        project="marcel",
        id=wandb_run_id,
        name=config["run_id"],
        config=config,
        resume="allow",
    )

    with open(wandb_json, "w") as fout:
        d = {"wandb_id": wandb_run.id}
        json.dump(d, fout)
    return wandb_run


if __name__ == "__main__":
    asyncio.run(main(parse_arguments()))

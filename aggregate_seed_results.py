#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def load_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_one_jsonl(path: Path):
    rows = load_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt_mean_std(values):
    mean, std = mean_std(values)
    if mean is None:
        return "n/a"
    return f"{mean:.2f} +/- {std:.2f}"


def candidate_roots(root: Path):
    roots = [root]
    nested_waterbirds = root / "waterbirds"
    if nested_waterbirds.exists():
        roots.append(nested_waterbirds)
    return roots


def first_existing_dir(roots, dirname: str):
    for base in roots:
        path = base / dirname
        if path.exists():
            return path
    return None


def collect_subgroup_table(root: Path, dataset_name: str, metric_dir: str, best_dir: str, top_key: str, wg_key: str, evals_by_method):
    roots = candidate_roots(root)
    rows = []
    for label, (run_name, eval_ids) in evals_by_method.items():
        best_top1_vals = []
        best_wg_vals = []
        uniform_top1_vals = []
        uniform_wg_vals = []
        greedy_top1_vals = []
        greedy_wg_vals = []
        used_eval_ids = []

        for eval_id in eval_ids:
            best_root = first_existing_dir(roots, f"eval_models_{run_name}_{best_dir}_eval{eval_id}")
            soup_root = first_existing_dir(roots, f"eval_models_{run_name}_{metric_dir}_eval{eval_id}")
            if best_root is None or soup_root is None:
                continue
            best_path = best_root / "individual_model_results.jsonl"
            uniform_path = soup_root / "uniform_soup_results.jsonl"
            greedy_path = soup_root / "greedy_soup_results.jsonl"

            if not (best_path.exists() and uniform_path.exists() and greedy_path.exists()):
                continue

            individuals = load_jsonl(best_path)
            uniform = load_one_jsonl(uniform_path)
            greedy = load_one_jsonl(greedy_path)

            best_top1_vals.append(max(item[top_key] for item in individuals) * 100.0)
            best_wg_vals.append(max(item[wg_key] for item in individuals) * 100.0)
            uniform_top1_vals.append(uniform[top_key] * 100.0)
            uniform_wg_vals.append(uniform[wg_key] * 100.0)
            greedy_top1_vals.append(greedy[top_key] * 100.0)
            greedy_wg_vals.append(greedy[wg_key] * 100.0)
            used_eval_ids.append(eval_id)

        rows.append(
            {
                "method": label,
                "dataset": dataset_name,
                "best_individual_avg": fmt_mean_std(best_top1_vals),
                "best_individual_wg": fmt_mean_std(best_wg_vals),
                "uniform_avg": fmt_mean_std(uniform_top1_vals),
                "uniform_wg": fmt_mean_std(uniform_wg_vals),
                "greedy_avg": fmt_mean_std(greedy_top1_vals),
                "greedy_wg": fmt_mean_std(greedy_wg_vals),
                "n_complete_evals": len(used_eval_ids),
                "eval_ids": used_eval_ids,
            }
        )

    return rows


def collect_natural_shift_status(root: Path):
    expected_bootstrap90 = [f"finetuned-models_bootstrap90_eval{i}" for i in range(5)]
    materialized_evals = sorted(path.name for path in root.glob("eval_bootstrap*_eval*"))
    available_metrics = sorted(path.name for path in root.glob("finetuned-models_bootstrap90_eval*/model_0_metrics.json"))
    return {
        "expected_bootstrap90_finetuned_dirs": expected_bootstrap90,
        "materialized_eval_dirs": materialized_evals,
        "bootstrap90_metric_files_detected": len(available_metrics),
    }


def render_table(rows):
    lines = [
        "| Method | Best Individual Avg | Best Individual WG | Uniform Avg | Uniform WG | Greedy Avg | Greedy WG | n | eval ids |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['best_individual_avg']} | {row['best_individual_wg']} | "
            f"{row['uniform_avg']} | {row['uniform_wg']} | {row['greedy_avg']} | {row['greedy_wg']} | "
            f"{row['n_complete_evals']} | {','.join(map(str, row['eval_ids'])) or 'none'} |"
        )
    return "\n".join(lines)


def render_latex_rows(rows):
    lines = []
    for row in rows:
        lines.append(
            f"{row['method']} & {row['best_individual_avg']} & {row['best_individual_wg']} & "
            f"{row['uniform_avg']} & {row['uniform_wg']} & {row['greedy_avg']} & {row['greedy_wg']} \\\\"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate repeated-seed results for the bootstrapped soups paper.")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("experiments"),
        help="Scratch directory containing eval_models_* and finetuned-models_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seed_result_summary.md"),
        help="Where to write the markdown summary.",
    )
    args = parser.parse_args()

    waterbirds_rows = collect_subgroup_table(
        root=args.scratch_root,
        dataset_name="Waterbirds",
        metric_dir="waterbirds",
        best_dir="top1_waterbirds",
        top_key="WaterbirdsTest_top1",
        wg_key="WaterbirdsTest_worst_group",
        evals_by_method={
            "VMS": ("nobootstrap", [0, 1, 2]),
            "BMS70": ("bootstrap70", [0, 1, 2]),
            "BMS80": ("bootstrap80", [0, 1, 2]),
            "BMS90": ("bootstrap90", [0, 1, 2]),
            "BMS100": ("bootstrap100", [0, 1, 2]),
        },
    )
    celeba_rows = collect_subgroup_table(
        root=args.scratch_root,
        dataset_name="CelebA",
        metric_dir="celeba_top1",
        best_dir="celeba_top1",
        top_key="CelebATest_top1",
        wg_key="CelebATest_worst_group",
        evals_by_method={
            "VMS": ("nobootstrap", [0, 1, 2]),
            "BMS70": ("bootstrap70", [0, 1, 2]),
            "BMS80": ("bootstrap80", [0, 1, 2]),
            "BMS90": ("bootstrap90", [0, 1, 2]),
            "BMS100": ("bootstrap100", [0, 1, 2]),
        },
    )
    natural_shift_status = collect_natural_shift_status(args.scratch_root)

    markdown = "\n".join(
        [
            "# Seed Result Summary",
            "",
            "This file aggregates the repeated-seed experiment folders currently materialized under the scratch workspace.",
            "Values are reported as `mean +/- std` across complete eval folders only.",
            "",
            "## Waterbirds",
            "",
            render_table(waterbirds_rows),
            "",
            "### Waterbirds LaTeX Rows",
            "",
            "```tex",
            render_latex_rows(waterbirds_rows),
            "```",
            "",
            "## CelebA",
            "",
            render_table(celeba_rows),
            "",
            "### CelebA LaTeX Rows",
            "",
            "```tex",
            render_latex_rows(celeba_rows),
            "```",
            "",
            "## Natural-Shift Status",
            "",
            f"- Bootstrap90 fine-tuned directories present: {', '.join(natural_shift_status['expected_bootstrap90_finetuned_dirs'])}",
            f"- Materialized natural-shift eval directories currently present: {', '.join(natural_shift_status['materialized_eval_dirs']) or 'none'}",
            f"- Bootstrap90 metric files detected across repeated fine-tuning runs: {natural_shift_status['bootstrap90_metric_files_detected']}",
            "- Interpretation: repeated fine-tuning runs exist for `bootstrap90`, but repeated natural-shift soup eval outputs are not all materialized on disk yet, so Table 1 cannot be converted cleanly to mean/std from the current saved artifacts alone.",
        ]
    )

    args.output.write_text(markdown)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

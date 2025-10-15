"""Analyze and visualize benchmark results."""

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_kernel_time_comparison(df: pd.DataFrame, output_dir: Path):
    df_valid = df[df["error"].isna()].copy()
    if df_valid.empty:
        return

    for workload in df_valid["workload_type"].unique():
        df_workload = df_valid[df_valid["workload_type"] == workload]

        configs = df_workload.groupby(["batch_size", "q_seq_len", "kv_seq_len"]).size()

        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5), squeeze=False)
        axes = axes.flatten()

        for idx, ((bs, q, kv), _) in enumerate(configs.items()):
            df_config = df_workload[
                (df_workload["batch_size"] == bs)
                & (df_workload["q_seq_len"] == q)
                & (df_workload["kv_seq_len"] == kv)
            ]

            pivot = df_config.pivot_table(
                index="framework",
                values="kernel_time_ms",
                aggfunc="mean",
            )

            pivot.plot(kind="bar", ax=axes[idx], rot=45, legend=False)
            axes[idx].set_title(f"B={bs}, Q={q}, KV={kv}")
            axes[idx].set_ylabel("Kernel Time (ms)")
            axes[idx].set_xlabel("Framework")
            axes[idx].grid(axis="y", alpha=0.3)

        fig.suptitle(f"Kernel Time - {workload}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = output_dir / f"kernel_time_{workload}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def plot_throughput_comparison(df: pd.DataFrame, output_dir: Path):
    df_valid = df[df["error"].isna()].copy()
    if df_valid.empty:
        return

    for workload in df_valid["workload_type"].unique():
        df_workload = df_valid[df_valid["workload_type"] == workload]

        configs = df_workload.groupby(["batch_size", "q_seq_len", "kv_seq_len"]).size()

        fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5), squeeze=False)
        axes = axes.flatten()

        for idx, ((bs, q, kv), _) in enumerate(configs.items()):
            df_config = df_workload[
                (df_workload["batch_size"] == bs)
                & (df_workload["q_seq_len"] == q)
                & (df_workload["kv_seq_len"] == kv)
            ]

            pivot = df_config.pivot_table(
                index="framework",
                values="throughput_tokens_per_sec",
                aggfunc="mean",
            )

            pivot.plot(kind="bar", ax=axes[idx], rot=45, legend=False)
            axes[idx].set_title(f"B={bs}, Q={q}, KV={kv}")
            axes[idx].set_ylabel("Throughput (tok/s)")
            axes[idx].set_xlabel("Framework")
            axes[idx].grid(axis="y", alpha=0.3)

        fig.suptitle(f"Throughput - {workload}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = output_dir / f"throughput_{workload}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def plot_memory_usage(df: pd.DataFrame, output_dir: Path):
    df_valid = df[df["error"].isna()].copy()
    if df_valid.empty:
        return

    for workload in df_valid["workload_type"].unique():
        df_workload = df_valid[df_valid["workload_type"] == workload]

        df_workload["seq_config"] = df_workload.apply(
            lambda r: f"Q={r['q_seq_len']},KV={r['kv_seq_len']}", axis=1
        )

        pivot = df_workload.pivot_table(
            index="framework",
            columns="seq_config",
            values="memory_allocated_gb",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(kind="bar", ax=ax, rot=45)
        ax.set_title(f"Memory Usage - {workload}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Memory (GB)")
        ax.set_xlabel("Framework")
        ax.legend(title="Config", bbox_to_anchor=(1.05, 1))
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"memory_{workload}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    df_valid = df[df["error"].isna()].copy()
    if df_valid.empty:
        return

    df_valid["workload_label"] = df_valid.apply(
        lambda r: f"{r['workload_type']}\nB={r['batch_size']},Q={r['q_seq_len']}", axis=1
    )

    pivot = df_valid.pivot_table(
        index="framework",
        columns="workload_label",
        values="throughput_tokens_per_sec",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Throughput (tok/s)"},
    )
    ax.set_title("Framework Performance Heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Framework")
    plt.tight_layout()

    output_path = output_dir / "heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary(df: pd.DataFrame, output_dir: Path):
    df_valid = df[df["error"].isna()].copy()

    if not df_valid.empty:
        summary = (
            df_valid.groupby(["framework", "workload_type"])
            .agg(
                {
                    "kernel_time_ms": ["mean", "std", "min", "max"],
                    "throughput_tokens_per_sec": ["mean", "std", "min", "max"],
                    "memory_allocated_gb": ["mean", "max"],
                }
            )
            .round(2)
        )

        summary_path = output_dir / "summary.csv"
        summary.to_csv(summary_path)
        print(f"Saved: {summary_path}")

        best = []
        for workload, group in df_valid.groupby("workload_type"):
            best_idx = group["throughput_tokens_per_sec"].idxmax()
            best_row = group.loc[best_idx]
            best.append(
                {
                    "workload_type": workload,
                    "best_framework": best_row["framework"],
                    "throughput": best_row["throughput_tokens_per_sec"],
                    "kernel_time_ms": best_row["kernel_time_ms"],
                }
            )

        best_df = pd.DataFrame(best)
        best_path = output_dir / "best_frameworks.csv"
        best_df.to_csv(best_path, index=False)
        print(f"Saved: {best_path}")

    df_errors = df[df["error"].notna()]
    if not df_errors.empty:
        error_summary = df_errors.groupby(["framework", "workload_type", "error"]).size().reset_index(name="count")
        error_path = output_dir / "errors.csv"
        error_summary.to_csv(error_path, index=False)
        print(f"Saved: {error_path}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(df)} | Success: {len(df_valid)} | Errors: {len(df_errors)}")
    print(f"Frameworks: {', '.join(sorted(df['framework'].unique()))}")
    print(f"Workloads: {', '.join(sorted(df['workload_type'].unique()))}")
    print(f"{'=' * 60}\n")


@click.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output-dir", default="./results/plots", type=click.Path())
@click.option("--skip-plots", is_flag=True)
def main(results_path, output_dir, skip_plots):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {results_path}")
    df = load_results(Path(results_path))

    print(f"Total: {len(df)} | Valid: {df['error'].isna().sum()} | Errors: {df['error'].notna().sum()}\n")

    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    if not skip_plots:
        print("Generating plots...")
        plot_kernel_time_comparison(df, output_dir)
        plot_throughput_comparison(df, output_dir)
        plot_memory_usage(df, output_dir)
        plot_heatmap(df, output_dir)

    print("\nGenerating summary...")
    generate_summary(df, output_dir)

    print(f"\nComplete! Outputs in: {output_dir}")


if __name__ == "__main__":
    main()

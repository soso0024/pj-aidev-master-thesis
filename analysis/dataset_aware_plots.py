#!/usr/bin/env python3
"""
Dataset-Aware Plotting Module

Contains visualization methods that analyze test generation
results by problem complexity.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAwarePlots:
    """Handles dataset-aware visualization methods for test result analysis."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        config_order: List[str],
    ):
        """Initialize with loaded data and ordering configurations."""
        self.data = data
        self.config_order = config_order
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self) -> pd.DataFrame:
        """Prepare pandas DataFrame from loaded data."""
        df = pd.DataFrame(self.data)

        # Create ordered categorical for consistent display
        df["config_type_display"] = pd.Categorical(
            df["config_type"], categories=self.config_order, ordered=True
        )

        return df

    def _format_model_name(self, model_name: str) -> str:
        """Format model name for display."""
        model_display_names = {
            "claude-3-5-haiku": "Claude 3.5 Haiku",
            "claude-opus-4-1": "Claude Opus 4.1",
            "claude-4-sonnet": "Claude 4 Sonnet",
            "claude-3-haiku": "Claude 3 Haiku",
        }
        return model_display_names.get(model_name, model_name.replace("-", " ").title())

    def create_all_plots(self, output_path: Path) -> None:
        """Create all dataset-aware visualization plots."""
        print("Creating dataset-aware visualizations...")

        self._plot_success_by_complexity(output_path)
        print("  ✓ Created success by complexity analysis")

        self._plot_cost_vs_coverage_by_model(output_path)
        print("  ✓ Created cost vs coverage by model analysis")

    def _plot_success_by_complexity(self, output_path: Path) -> None:
        """Plot success rate by problem complexity level."""
        if "complexity_level" not in self.df.columns:
            print(
                "Warning: complexity_level not found in data. Skipping complexity analysis."
            )
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Success rate by complexity level
        complexity_order = ["simple", "medium", "complex"]
        complexity_data = self.df[self.df["complexity_level"].isin(complexity_order)]

        # Ensure proper ordering
        complexity_data["complexity_level"] = pd.Categorical(
            complexity_data["complexity_level"],
            categories=complexity_order,
            ordered=True,
        )

        # Heatmap of success rate
        if not complexity_data.empty:
            pivot_heatmap = complexity_data.pivot_table(
                values="success",
                index="complexity_level",
                columns="config_type_display",
                aggfunc="mean",
                fill_value=0,
                observed=False,
            )

            if not pivot_heatmap.empty:
                heatmap = sns.heatmap(
                    pivot_heatmap,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1,
                    ax=ax,
                    cbar_kws={"label": "Success Rate"},
                    annot_kws={"fontsize": 26, "fontweight": "bold"},
                )
                # Increase colorbar label and tick label sizes
                cbar = heatmap.collections[0].colorbar
                cbar.ax.tick_params(labelsize=14)
                cbar.set_label("Success Rate", size=16)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No classified complexity data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No classified complexity data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("Configuration Type", fontsize=20, fontweight="bold")
        ax.set_ylabel("Problem Complexity Level", fontsize=20, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=16)

        plt.tight_layout()
        plt.savefig(
            output_path / "6_success_by_complexity.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_cost_vs_coverage_by_model(self, output_path: Path) -> None:
        """Plot cost vs coverage scatter plots for each model."""
        if "model" not in self.df.columns:
            print(
                "Warning: model information not found in data. Skipping cost vs coverage by model analysis."
            )
            return

        # Get unique models, excluding 'unknown'
        models = [model for model in self.df["model"].unique() if model != "unknown"]

        if not models:
            print(
                "Warning: No valid model data found. Skipping cost vs coverage by model analysis."
            )
            return

        # If only one model, create a single focused plot
        if len(models) == 1:
            model = models[0]
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            model_data = self.df[self.df["model"] == model]

            # Color map for different configurations
            config_colors = {
                "basic": "#1f77b4",
                "ast": "#ff7f0e",
                "docstring": "#2ca02c",
                "docstring_ast": "#d62728",
                "ast-fix": "#9467bd",
            }

            # Calculate average cost and coverage by configuration
            avg_stats = (
                model_data.groupby("config_type_display", observed=False)
                .agg(
                    {
                        "total_cost_usd": "mean",
                        "code_coverage_percent": "mean",
                        "success": "mean",  # success rate
                        "config_type": "count",  # for sample size
                    }
                )
                .reset_index()
            )

            # Calculate efficiency score (coverage per $0.001)
            avg_stats["efficiency_score"] = avg_stats["code_coverage_percent"] / (
                avg_stats["total_cost_usd"] * 1000
            )

            # Plot scatter points for each configuration
            for _, row in avg_stats.iterrows():
                config = row["config_type_display"]
                cost = row["total_cost_usd"]
                coverage = row["code_coverage_percent"]
                count = row["config_type"]
                success_rate = row["success"] * 100
                efficiency = row["efficiency_score"]

                color = config_colors.get(config, "#888888")
                ax.scatter(
                    cost,
                    coverage,
                    color=color,
                    s=150,
                    alpha=0.8,
                    label=config,
                    edgecolors="black",
                    linewidths=1,
                )

                # Add text annotation with configuration name
                # Position on the left for 'ast' configuration only for claude-3-5-haiku
                config_x_offset = (
                    -8 if (config == "ast" and model == "claude-3-5-haiku") else 8
                )
                config_h_align = (
                    "right"
                    if (config == "ast" and model == "claude-3-5-haiku")
                    else "left"
                )
                ax.annotate(
                    config,
                    (cost, coverage),
                    xytext=(config_x_offset, 8),
                    textcoords="offset points",
                    fontsize=18,
                    fontweight="bold",
                    alpha=0.9,
                    ha=config_h_align,
                )

                # Add efficiency and success rate annotations
                # Position text on the left for 'ast' configuration only for claude-3-5-haiku, right for others
                is_ast_left = config == "ast" and model == "claude-3-5-haiku"
                x_offset = -8 if is_ast_left else 8
                h_align = "right" if is_ast_left else "left"
                # Adjust y_offset for better spacing, especially for docstring_ast
                y_offset = -30
                ax.annotate(
                    f"CCE: {efficiency:.1f}\nSuccess: {success_rate:.1f}%",
                    (cost, coverage),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    fontsize=16,
                    fontweight="bold",
                    alpha=0.7,
                    ha=h_align,
                )

            # Formatting
            ax.set_xlabel("Average Total Cost (USD)", fontsize=20, fontweight="bold")
            ax.set_ylabel("Average Code Coverage (%)", fontsize=20, fontweight="bold")
            # Title removed for paper publication
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.grid(True, alpha=0.3)

            # Set axis limits based on actual data range with some padding
            if not avg_stats.empty:
                cost_min = avg_stats["total_cost_usd"].min()
                cost_max = avg_stats["total_cost_usd"].max()
                coverage_min = avg_stats["code_coverage_percent"].min()
                coverage_max = avg_stats["code_coverage_percent"].max()

                # Add padding (10% of range) for better visualization
                cost_range = cost_max - cost_min
                coverage_range = coverage_max - coverage_min

                cost_padding = max(
                    cost_range * 0.1, 0.001
                )  # Minimum padding for small ranges
                coverage_padding = max(
                    coverage_range * 0.1, 2
                )  # Minimum 2% padding for coverage

                ax.set_xlim(max(0, cost_min - cost_padding), cost_max + cost_padding)
                ax.set_ylim(
                    max(0, coverage_min - coverage_padding),
                    min(100, coverage_max + coverage_padding),
                )
            else:
                ax.set_xlim(left=0)
                ax.set_ylim(0, 100)

            # Add legend
            ax.legend(loc="upper right", fontsize=16)

            plt.tight_layout()

        else:
            # Multiple models - use subplot layout
            n_models = len(models)
            if n_models == 2:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            elif n_models <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.ravel()
            else:
                # For more than 4 models, use a larger grid
                cols = 3
                rows = (n_models + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
                axes = axes.ravel()

            # Color map for different configurations
            config_colors = {
                "basic": "#1f77b4",
                "ast": "#ff7f0e",
                "docstring": "#2ca02c",
                "docstring_ast": "#d62728",
                "ast-fix": "#9467bd",
            }

            for idx, model in enumerate(models):
                if idx >= len(axes):
                    break

                ax = axes[idx]
                model_data = self.df[self.df["model"] == model]

                # Calculate average cost and coverage by configuration
                avg_stats = (
                    model_data.groupby("config_type_display", observed=False)
                    .agg(
                        {
                            "total_cost_usd": "mean",
                            "code_coverage_percent": "mean",
                            "success": "mean",  # success rate
                            "config_type": "count",  # for sample size
                        }
                    )
                    .reset_index()
                )

                # Calculate efficiency score (coverage per $0.001)
                avg_stats["efficiency_score"] = avg_stats["code_coverage_percent"] / (
                    avg_stats["total_cost_usd"] * 1000
                )

                # Plot scatter points for each configuration
                for _, row in avg_stats.iterrows():
                    config = row["config_type_display"]
                    cost = row["total_cost_usd"]
                    coverage = row["code_coverage_percent"]
                    count = row["config_type"]
                    success_rate = row["success"] * 100
                    efficiency = row["efficiency_score"]

                    color = config_colors.get(config, "#888888")
                    ax.scatter(
                        cost,
                        coverage,
                        color=color,
                        s=100,
                        alpha=0.8,
                        label=config,
                        edgecolors="black",
                        linewidths=0.5,
                    )

                    # Add text annotation with configuration name
                    # Position on the left for 'ast' configuration only for claude-3-5-haiku
                    config_x_offset = (
                        -8 if (config == "ast" and model == "claude-3-5-haiku") else 5
                    )
                    config_h_align = (
                        "right"
                        if (config == "ast" and model == "claude-3-5-haiku")
                        else "left"
                    )
                    ax.annotate(
                        config,
                        (cost, coverage),
                        xytext=(config_x_offset, 5),
                        textcoords="offset points",
                        fontsize=18,
                        fontweight="bold",
                        alpha=0.8,
                        ha=config_h_align,
                    )

                    # Add efficiency and success rate annotations
                    # Position text on the left for 'ast' configuration only for claude-3-5-haiku
                    is_ast_left = config == "ast" and model == "claude-3-5-haiku"
                    x_offset = -8 if is_ast_left else 5
                    h_align = "right" if is_ast_left else "left"
                    ax.annotate(
                        f"CCE: {efficiency:.1f}\nSuccess: {success_rate:.1f}%",
                        (cost, coverage),
                        xytext=(x_offset, -20),
                        textcoords="offset points",
                        fontsize=16,
                        fontweight="bold",
                        alpha=0.7,
                        ha=h_align,
                    )

                # Formatting
                ax.set_xlabel(
                    "Average Total Cost (USD)", fontsize=20, fontweight="bold"
                )
                ax.set_ylabel(
                    "Average Code Coverage (%)", fontsize=20, fontweight="bold"
                )
                # Individual subplot titles removed for paper publication
                ax.tick_params(axis="both", which="major", labelsize=16)
                ax.grid(True, alpha=0.3)

                # Set axis limits based on actual data range with some padding
                if not avg_stats.empty:
                    cost_min = avg_stats["total_cost_usd"].min()
                    cost_max = avg_stats["total_cost_usd"].max()
                    coverage_min = avg_stats["code_coverage_percent"].min()
                    coverage_max = avg_stats["code_coverage_percent"].max()

                    # Add padding (10% of range) for better visualization
                    cost_range = cost_max - cost_min
                    coverage_range = coverage_max - coverage_min

                    cost_padding = max(
                        cost_range * 0.1, 0.001
                    )  # Minimum padding for small ranges
                    coverage_padding = max(
                        coverage_range * 0.1, 2
                    )  # Minimum 2% padding for coverage

                    ax.set_xlim(
                        max(0, cost_min - cost_padding), cost_max + cost_padding
                    )
                    ax.set_ylim(
                        max(0, coverage_min - coverage_padding),
                        min(100, coverage_max + coverage_padding),
                    )
                else:
                    ax.set_xlim(left=0)
                    ax.set_ylim(0, 100)

                # Add legend
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)

            # Hide empty subplots if any
            for idx in range(n_models, len(axes)):
                axes[idx].set_visible(False)

            # Overall title removed for paper publication

            plt.tight_layout()

        plt.savefig(
            output_path / "7_cost_vs_coverage_by_model.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

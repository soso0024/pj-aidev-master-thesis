#!/usr/bin/env python3
"""
Cross-Model Comparison Plots Module

Creates visualizations comparing multiple models across various metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns


class CrossModelPlots:
    """Handles creation of cross-model comparison visualizations."""

    def __init__(
        self, model_data: Dict[str, List[Dict[str, Any]]], config_order: List[str]
    ):
        """
        Initialize with data from multiple models.

        Args:
            model_data: Dictionary mapping model names to their data lists
            config_order: Order of configurations for consistent plotting
        """
        self.model_data = model_data
        self.config_order = config_order
        self.model_names = sorted(model_data.keys())

        # Detect if this is HumanEvalPack data (bug detection task)
        self.is_humanevalpack = self._detect_humanevalpack_data()

        # Set up plotting style
        plt.style.use("seaborn-v0_8-darkgrid")

        # Use custom color palette for better visual clarity
        base_colors = ["#FF4B00", "#005AFF", "#03AF7A", "#4DC4FF", "#F6AA00"]

        # Extend colors if we have more models than base colors
        if len(self.model_names) <= len(base_colors):
            self.colors = base_colors[: len(self.model_names)]
        else:
            # Repeat and extend the color palette
            self.colors = (
                base_colors * ((len(self.model_names) // len(base_colors)) + 1)
            )[: len(self.model_names)]

        # Configuration-specific colors (lighter shades for visual distinction)
        self.config_colors = {
            "basic": "#FFB380",  # Light orange
            "ast": "#80ADFF",  # Light blue
            "docstring": "#81D7BC",  # Light green
            "docstring_ast": "#A6E2FF",  # Light cyan
        }

        # Configuration display name mapping (Japanese)
        self.config_display_names = {
            "basic": "基本プロンプト",
            "ast": "基本 + AST",
            "docstring": "基本 + Docstring",
            "docstring_ast": "基本 + Docstring + AST",
        }

    def _detect_humanevalpack_data(self) -> bool:
        """Detect if the data is from HumanEvalPack dataset."""
        for model_data in self.model_data.values():
            if model_data and any("buggy_failure_type" in d for d in model_data):
                return True
        return False

    def _is_success(self, record: Dict[str, Any]) -> bool:
        """
        Determine if a record represents a successful test.

        For HumanEvalPack: Success = detecting the bug (assertion failure)
        For HumanEval: Success = test passed
        """
        if self.is_humanevalpack:
            # For HumanEvalPack, success means detecting the bug (AssertionError)
            failure_type = record.get("buggy_failure_type", "")
            return failure_type == "assertion"
        else:
            # For regular HumanEval, success means test passed
            return record.get("test_passed", False)

    def _clean_model_name(self, model_name: str) -> str:
        """
        Clean model name for display in legends.
        Removes common prefixes like 'humanevalpack_'.
        """
        # Remove common prefixes
        prefixes_to_remove = ["humanevalpack_", "humaneval_", "generated_tests_"]
        clean_name = model_name
        for prefix in prefixes_to_remove:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix) :]
        return clean_name

    def create_all_plots(self, output_dir: Path) -> None:
        """Create all cross-model comparison plots."""
        print("\n📊 Creating cross-model comparison plots...")

        # Set global plotting parameters for consistent, professional appearance
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["axes.linewidth"] = 1.2
        plt.rcParams["grid.color"] = "#CCCCCC"
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.linewidth"] = 0.5
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 10
        
        # Set Japanese font for all plots
        plt.rcParams["font.sans-serif"] = [
            "Hiragino Sans",
            "Yu Gothic",
            "Meiryo",
            "Takao",
            "IPAexGothic",
            "IPAPGothic",
            "VL PGothic",
            "Noto Sans CJK JP",
        ] + plt.rcParams["font.sans-serif"]

        self.plot_bug_detection_rate_comparison(output_dir)
        self.plot_test_generation_success_rate(output_dir)
        self.plot_cost_comparison(output_dir)
        self.plot_coverage_comparison(output_dir)
        self.plot_efficiency_comparison(output_dir)
        self.plot_fix_attempts_comparison(output_dir)
        self.plot_token_usage_comparison(output_dir)
        self.plot_cost_efficiency_scatter(output_dir)
        self.plot_model_config_heatmap(output_dir)
        self.plot_overall_rankings(output_dir)

        print("✅ Cross-model comparison plots created")

    def plot_bug_detection_rate_comparison(self, output_dir: Path) -> None:
        """Plot bug detection rates across all models and configurations."""
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.config_order))
        width = 0.8 / len(self.model_names)

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            success_rates = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                if config_data:
                    success_count = sum(1 for d in config_data if self._is_success(d))
                    success_rate = (success_count / len(config_data)) * 100
                    success_rates.append(success_rate)
                else:
                    success_rates.append(0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                success_rates,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars (larger font for paper)
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 1.5,
                        f"{rate:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontweight="bold", fontsize=16)
        ax.set_ylabel(
            "Bug Detection Rate (%)" if self.is_humanevalpack else "Success Rate (%)",
            fontweight="bold",
            fontsize=16,
        )
        title = (
            "Bug Detection Rate Comparison Across Models"
            if self.is_humanevalpack
            else "Test Success Rate Comparison Across Models"
        )
        ax.set_title(title, fontweight="bold", fontsize=18, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            framealpha=0.95,
            fontsize=13,
            ncol=1,
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_1a_bug_detection_rate.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_test_generation_success_rate(self, output_dir: Path) -> None:
        """Plot test generation success rates (evaluation_success) across all models and configurations."""
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.config_order))
        width = 0.8 / len(self.model_names)

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            success_rates = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                if config_data:
                    # Use evaluation_success for test generation success
                    success_count = sum(
                        1 for d in config_data if d.get("evaluation_success", False)
                    )
                    success_rate = (success_count / len(config_data)) * 100
                    success_rates.append(success_rate)
                else:
                    success_rates.append(0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                success_rates,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars (numbers only, no % symbol)
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 1.5,
                        f"{rate:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

        ax.set_xlabel("プロンプト構成", fontweight="bold", fontsize=16)
        ax.set_ylabel("生成成功率（％）", fontweight="bold", fontsize=16)
        # タイトルを削除
        # ax.set_title("Test Generation Success Rate Comparison Across Models", fontweight="bold", fontsize=18, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            framealpha=0.95,
            fontsize=13,
            ncol=1,
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_1b_test_generation_success_rate.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_cost_comparison(self, output_dir: Path) -> None:
        """Plot cost comparison across models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Average cost per problem
        model_avg_costs = {}
        for model_name in self.model_names:
            data = self.model_data[model_name]
            costs = [d.get("total_cost", 0) for d in data if d.get("total_cost", 0) > 0]
            model_avg_costs[model_name] = np.mean(costs) if costs else 0

        # Plot 1: Average cost per model
        bars1 = ax1.bar(
            range(len(self.model_names)),
            [model_avg_costs[m] for m in self.model_names],
            color=self.colors,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax1.set_xlabel("Model", fontweight="bold", fontsize=14)
        ax1.set_ylabel("Average Cost ($)", fontweight="bold", fontsize=14)
        ax1.set_title(
            "Average Cost per Problem by Model", fontweight="bold", fontsize=15
        )
        ax1.set_xticks(range(len(self.model_names)))
        ax1.set_xticklabels(
            [self._clean_model_name(m) for m in self.model_names],
            rotation=45,
            ha="right",
            fontsize=12,
        )
        ax1.tick_params(axis="y", labelsize=12)
        ax1.grid(True, alpha=0.3, axis="y", linewidth=0.8)

        # Add value labels
        for bar, model in zip(bars1, self.model_names):
            height = bar.get_height()
            if height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"${height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )

        # Plot 2: Cost by configuration
        x = np.arange(len(self.config_order))
        width = 0.8 / len(self.model_names)

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            config_costs = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                costs = [
                    d.get("total_cost", 0)
                    for d in config_data
                    if d.get("total_cost", 0) > 0
                ]
                config_costs.append(np.mean(costs) if costs else 0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            ax2.bar(
                x + offset,
                config_costs,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

        ax2.set_xlabel("Configuration", fontweight="bold", fontsize=14)
        ax2.set_ylabel("Average Cost ($)", fontweight="bold", fontsize=14)
        ax2.set_title("Average Cost by Configuration", fontweight="bold", fontsize=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=12,
        )
        ax2.tick_params(axis="y", labelsize=12)
        # Place legend outside the plot area (right side)
        ax2.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.95, fontsize=11
        )
        ax2.grid(True, alpha=0.3, axis="y", linewidth=0.8)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_2_cost_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_coverage_comparison(self, output_dir: Path) -> None:
        """Plot code coverage comparison across models (C0 and C1 as separate files)."""
        x = np.arange(len(self.config_order))
        width = 0.8 / len(self.model_names)

        # ==================== Plot C0: Statement Coverage ====================
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            c0_rates = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                # Use C0 coverage (statement coverage)
                coverages = [d.get("code_coverage_c0_percent", 0) for d in config_data]
                c0_rates.append(np.mean(coverages) if coverages else 0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                c0_rates,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars (larger font for paper)
            for bar, rate in zip(bars, c0_rates):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 1.5,
                        f"{rate:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontweight="bold", fontsize=16)
        ax.set_ylabel("C0 Statement Coverage (%)", fontweight="bold", fontsize=16)
        ax.set_title(
            "C0: Statement Coverage Comparison Across Models",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            framealpha=0.95,
            fontsize=13,
            ncol=1,
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_3a_c0_coverage.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # ==================== Plot C1: Branch Coverage ====================
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            c1_rates = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                # Use C1 coverage (branch coverage)
                coverages = [d.get("code_coverage_c1_percent", 0) for d in config_data]
                c1_rates.append(np.mean(coverages) if coverages else 0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                c1_rates,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars (larger font for paper)
            for bar, rate in zip(bars, c1_rates):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 1.5,
                        f"{rate:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontweight="bold", fontsize=16)
        ax.set_ylabel("C1 Branch Coverage (%)", fontweight="bold", fontsize=16)
        ax.set_title(
            "C1: Branch Coverage Comparison Across Models",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            framealpha=0.95,
            fontsize=13,
            ncol=1,
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_3b_c1_coverage.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_efficiency_comparison(self, output_dir: Path) -> None:
        """Plot efficiency metrics (success rate / cost) across models."""
        fig, ax = plt.subplots(figsize=(12, 7))

        efficiency_scores = {}
        for model_name in self.model_names:
            data = self.model_data[model_name]
            success_count = sum(1 for d in data if self._is_success(d))
            total_cost = sum(d.get("total_cost", 0) for d in data)

            if total_cost > 0:
                # Efficiency = (Success Rate) / Cost
                success_rate = (success_count / len(data)) * 100 if data else 0
                efficiency = success_rate / total_cost
                efficiency_scores[model_name] = efficiency
            else:
                efficiency_scores[model_name] = 0

        bars = ax.bar(
            range(len(self.model_names)),
            [efficiency_scores[m] for m in self.model_names],
            color=self.colors,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax.set_xlabel("Model", fontweight="bold", fontsize=16)
        ax.set_ylabel(
            "Efficiency Score (Success Rate % / $)", fontweight="bold", fontsize=16
        )
        ax.set_title(
            "Cost Efficiency Comparison Across Models\n(Higher is Better)",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.set_xticks(range(len(self.model_names)))
        ax.set_xticklabels(
            [self._clean_model_name(m) for m in self.model_names],
            rotation=45,
            ha="right",
            fontsize=13,
        )
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)

        # Add value labels
        for bar, model in zip(bars, self.model_names):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_4_efficiency.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_fix_attempts_comparison(self, output_dir: Path) -> None:
        """Plot fix attempts comparison across models."""
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.config_order))
        width = 0.8 / len(self.model_names)

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            avg_attempts = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                attempts = [d.get("fix_attempts", 0) for d in config_data]
                avg_attempts.append(np.mean(attempts) if attempts else 0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                avg_attempts,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars (larger font for paper)
            for bar, attempts in zip(bars, avg_attempts):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.05,
                        f"{attempts:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontweight="bold", fontsize=16)
        ax.set_ylabel("Average Fix Attempts", fontweight="bold", fontsize=16)
        ax.set_title(
            "Fix Attempts Comparison Across Models",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.95, fontsize=13
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_5_fix_attempts.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_token_usage_comparison(self, output_dir: Path) -> None:
        """Plot token usage comparison across models (input, output, and total tokens as separate files)."""
        x = np.arange(len(self.config_order))
        width = 0.8 / len(self.model_names)

        # ==================== Plot Input Tokens ====================
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            input_tokens = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                tokens = [
                    d.get("total_input_tokens", 0)
                    for d in config_data
                    if d.get("total_input_tokens", 0) > 0
                ]
                input_tokens.append(np.mean(tokens) if tokens else 0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                input_tokens,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars
            for bar, tokens in zip(bars, input_tokens):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(tokens):,}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontweight="bold", fontsize=16)
        ax.set_ylabel("Average Input Tokens", fontweight="bold", fontsize=16)
        ax.set_title(
            "Input Token Usage Comparison Across Models",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.95, fontsize=13
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_6a_input_tokens.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # ==================== Plot Output Tokens ====================
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            output_tokens = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                tokens = [
                    d.get("total_output_tokens", 0)
                    for d in config_data
                    if d.get("total_output_tokens", 0) > 0
                ]
                output_tokens.append(np.mean(tokens) if tokens else 0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                output_tokens,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars
            for bar, tokens in zip(bars, output_tokens):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(tokens):,}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontweight="bold", fontsize=16)
        ax.set_ylabel("Average Output Tokens", fontweight="bold", fontsize=16)
        ax.set_title(
            "Output Token Usage Comparison Across Models",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.95, fontsize=13
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_6b_output_tokens.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # ==================== Plot Total Tokens ====================
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]
            total_tokens = []

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                tokens = [
                    d.get("total_tokens", 0)
                    for d in config_data
                    if d.get("total_tokens", 0) > 0
                ]
                total_tokens.append(np.mean(tokens) if tokens else 0)

            offset = (i - len(self.model_names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                total_tokens,
                width,
                label=self._clean_model_name(model_name),
                color=self.colors[i],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=1.2,
            )

            # Add value labels on bars
            for bar, tokens in zip(bars, total_tokens):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(tokens):,}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

        ax.set_xlabel("Configuration", fontweight="bold", fontsize=16)
        ax.set_ylabel("Average Total Tokens", fontweight="bold", fontsize=16)
        ax.set_title(
            "Total Token Usage Comparison Across Models",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [self.config_display_names.get(c, c) for c in self.config_order],
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.95, fontsize=13
        )
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.8)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_6c_total_tokens.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_cost_efficiency_scatter(self, output_dir: Path) -> None:
        """Scatter plot: Cost vs Success Rate for all models."""
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, model_name in enumerate(self.model_names):
            data = self.model_data[model_name]

            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                if not config_data:
                    continue

                success_count = sum(1 for d in config_data if self._is_success(d))
                success_rate = (success_count / len(config_data)) * 100
                avg_cost = np.mean([d.get("total_cost", 0) for d in config_data])

                marker = ["o", "s", "^", "D"][self.config_order.index(config)]
                ax.scatter(
                    avg_cost,
                    success_rate,
                    s=180,
                    alpha=0.75,
                    color=self.colors[i],
                    marker=marker,
                    label=(
                        f"{self._clean_model_name(model_name)} - {config}"
                        if config == self.config_order[0]
                        else ""
                    ),
                    edgecolors="#333333",
                    linewidth=1.2,
                )

        ax.set_xlabel("Average Cost ($)", fontweight="bold", fontsize=16)
        ax.set_ylabel(
            "Bug Detection Rate (%)" if self.is_humanevalpack else "Success Rate (%)",
            fontweight="bold",
            fontsize=16,
        )
        title = (
            "Cost vs Bug Detection Rate Trade-off"
            if self.is_humanevalpack
            else "Cost vs Success Rate Trade-off"
        )
        ax.set_title(
            f"{title}\n(Top-right is best: High success, Low cost)",
            fontweight="bold",
            fontsize=18,
            pad=20,
        )
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        # Place legend outside the plot area (right side)
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.95, fontsize=11
        )

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_7_cost_vs_success.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_model_config_heatmap(self, output_dir: Path) -> None:
        """Heatmap of success rates for model-configuration combinations."""
        fig, ax = plt.subplots(figsize=(12, len(self.model_names) * 0.8 + 2))

        # Create matrix of success rates
        matrix = []
        for model_name in self.model_names:
            data = self.model_data[model_name]
            row = []
            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                if config_data:
                    success_count = sum(1 for d in config_data if self._is_success(d))
                    success_rate = (success_count / len(config_data)) * 100
                    row.append(success_rate)
                else:
                    row.append(0)
            matrix.append(row)

        # Create heatmap with custom colormap for better visibility
        # Use a colormap that goes from red (low) -> yellow (mid) -> green (high)
        from matplotlib.colors import LinearSegmentedColormap

        colors_map = [
            "#FF4B00",
            "#F6AA00",
            "#03AF7A",
        ]  # Red -> Yellow -> Green from our palette
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("custom", colors_map, N=n_bins)

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.config_order)))
        ax.set_yticks(np.arange(len(self.model_names)))
        ax.set_xticklabels(self.config_order)
        ax.set_yticklabels([self._clean_model_name(m) for m in self.model_names])

        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations with better contrast
        for i in range(len(self.model_names)):
            for j in range(len(self.config_order)):
                value = matrix[i][j]
                # Use white text for darker backgrounds, black for lighter
                text_color = "white" if value > 60 or value < 30 else "black"
                text = ax.text(
                    j,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=13,
                )

        title_text = (
            "Bug Detection Rate Heatmap: Models vs Configurations"
            if self.is_humanevalpack
            else "Success Rate Heatmap: Models vs Configurations"
        )
        ax.set_title(title_text, fontweight="bold", fontsize=18, pad=20)
        ax.tick_params(axis="both", labelsize=13)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            "Bug Detection Rate (%)" if self.is_humanevalpack else "Success Rate (%)",
            fontsize=14,
            fontweight="bold",
        )
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_8_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_overall_rankings(self, output_dir: Path) -> None:
        """Plot overall rankings of models based on multiple metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        metrics = {}
        for model_name in self.model_names:
            data = self.model_data[model_name]

            # Success rate
            success_count = sum(1 for d in data if self._is_success(d))
            success_rate = (success_count / len(data)) * 100 if data else 0

            # Average cost
            costs = [d.get("total_cost", 0) for d in data if d.get("total_cost", 0) > 0]
            avg_cost = np.mean(costs) if costs else 0

            # Average coverage (include all records for consistency)
            coverages = [d.get("coverage_percentage", 0) for d in data]
            avg_coverage = np.mean(coverages) if coverages else 0

            # Average fix attempts
            attempts = [d.get("fix_attempts", 0) for d in data]
            avg_attempts = np.mean(attempts) if attempts else 0

            metrics[model_name] = {
                "success_rate": success_rate,
                "avg_cost": avg_cost,
                "avg_coverage": avg_coverage,
                "avg_attempts": avg_attempts,
            }

        # Plot 1: Success Rate
        sorted_models = sorted(
            self.model_names, key=lambda m: metrics[m]["success_rate"], reverse=True
        )
        colors_sorted = [self.colors[self.model_names.index(m)] for m in sorted_models]
        ax1.barh(
            range(len(sorted_models)),
            [metrics[m]["success_rate"] for m in sorted_models],
            color=colors_sorted,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax1.set_yticks(range(len(sorted_models)))
        ax1.set_yticklabels(
            [self._clean_model_name(m) for m in sorted_models], fontsize=12
        )
        label_text = (
            "Bug Detection Rate (%)" if self.is_humanevalpack else "Success Rate (%)"
        )
        ax1.set_xlabel(label_text, fontweight="bold", fontsize=13)
        title_text = (
            "Ranking by Bug Detection Rate"
            if self.is_humanevalpack
            else "Ranking by Success Rate"
        )
        ax1.set_title(title_text, fontweight="bold", fontsize=14)
        ax1.tick_params(axis="x", labelsize=12)
        ax1.grid(True, alpha=0.3, axis="x", linewidth=0.8)
        for i, model in enumerate(sorted_models):
            ax1.text(
                metrics[model]["success_rate"],
                i,
                f" {metrics[model]['success_rate']:.1f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        # Plot 2: Cost (lower is better)
        sorted_models = sorted(self.model_names, key=lambda m: metrics[m]["avg_cost"])
        colors_sorted = [self.colors[self.model_names.index(m)] for m in sorted_models]
        ax2.barh(
            range(len(sorted_models)),
            [metrics[m]["avg_cost"] for m in sorted_models],
            color=colors_sorted,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax2.set_yticks(range(len(sorted_models)))
        ax2.set_yticklabels(
            [self._clean_model_name(m) for m in sorted_models], fontsize=12
        )
        ax2.set_xlabel("Average Cost ($)", fontweight="bold", fontsize=13)
        ax2.set_title(
            "Ranking by Cost (Lower is Better)", fontweight="bold", fontsize=14
        )
        ax2.tick_params(axis="x", labelsize=12)
        ax2.grid(True, alpha=0.3, axis="x", linewidth=0.8)
        for i, model in enumerate(sorted_models):
            ax2.text(
                metrics[model]["avg_cost"],
                i,
                f" ${metrics[model]['avg_cost']:.4f}",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        # Plot 3: Coverage
        sorted_models = sorted(
            self.model_names, key=lambda m: metrics[m]["avg_coverage"], reverse=True
        )
        colors_sorted = [self.colors[self.model_names.index(m)] for m in sorted_models]
        ax3.barh(
            range(len(sorted_models)),
            [metrics[m]["avg_coverage"] for m in sorted_models],
            color=colors_sorted,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax3.set_yticks(range(len(sorted_models)))
        ax3.set_yticklabels(
            [self._clean_model_name(m) for m in sorted_models], fontsize=12
        )
        ax3.set_xlabel("Average Coverage (%)", fontweight="bold", fontsize=13)
        ax3.set_title("Ranking by Code Coverage", fontweight="bold", fontsize=14)
        ax3.tick_params(axis="x", labelsize=12)
        ax3.grid(True, alpha=0.3, axis="x", linewidth=0.8)
        for i, model in enumerate(sorted_models):
            ax3.text(
                metrics[model]["avg_coverage"],
                i,
                f" {metrics[model]['avg_coverage']:.1f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        # Plot 4: Fix Attempts (lower is better)
        sorted_models = sorted(
            self.model_names, key=lambda m: metrics[m]["avg_attempts"]
        )
        colors_sorted = [self.colors[self.model_names.index(m)] for m in sorted_models]
        ax4.barh(
            range(len(sorted_models)),
            [metrics[m]["avg_attempts"] for m in sorted_models],
            color=colors_sorted,
            alpha=0.85,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax4.set_yticks(range(len(sorted_models)))
        ax4.set_yticklabels(
            [self._clean_model_name(m) for m in sorted_models], fontsize=12
        )
        ax4.set_xlabel("Average Fix Attempts", fontweight="bold", fontsize=13)
        ax4.set_title(
            "Ranking by Fix Attempts (Lower is Better)", fontweight="bold", fontsize=14
        )
        ax4.tick_params(axis="x", labelsize=12)
        ax4.grid(True, alpha=0.3, axis="x", linewidth=0.8)
        for i, model in enumerate(sorted_models):
            ax4.text(
                metrics[model]["avg_attempts"],
                i,
                f" {metrics[model]['avg_attempts']:.1f}",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

        plt.suptitle(
            "Overall Model Rankings by Key Metrics",
            fontweight="bold",
            fontsize=18,
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "cross_model_9_rankings.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

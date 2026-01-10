#!/usr/bin/env python3
"""
Cross-Model Test Results Analysis Tool

Analyzes and compares test results across multiple LLM models.
Loads data from multiple model directories and creates comprehensive
comparison visualizations.

Usage:
    python run_cross_model_analysis.py --models-dir data --output-dir cross_model_vizs

The script will automatically detect all model directories in the specified path
that match the pattern: generated_tests_*_{model_name}
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Any

# Import our modular components
from analysis.data_loader import DataLoader
from analysis.cross_model_plots import CrossModelPlots


class CrossModelAnalyzer:
    """Main orchestrator for cross-model analysis."""

    def __init__(
        self,
        models_dir: str = "data",
        dataset_path: str = "dataset/HumanEval.jsonl",
        models_config_path: str = "models_config.json",
    ):
        """
        Initialize the cross-model analyzer.

        Args:
            models_dir: Directory containing model result directories
            dataset_path: Path to the dataset file
            models_config_path: Path to models configuration JSON
        """
        self.models_dir = Path(models_dir)
        self.dataset_path = dataset_path
        self.models_config_path = models_config_path
        self.config_order = ["basic", "ast", "docstring", "docstring_ast"]

        # Load models configuration
        self.models_config = self._load_models_config()

        # Storage for loaded data
        self.model_data: Dict[str, List[Dict[str, Any]]] = {}
        self.cross_model_plots = None

        # Detect if this is HumanEvalPack data
        self.is_humanevalpack = False

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

    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from JSON file."""
        if os.path.exists(self.models_config_path):
            with open(self.models_config_path, "r") as f:
                return json.load(f)
        return {"models": {}}

    def _detect_model_directories(self) -> Dict[str, Path]:
        """
        Detect all model result directories in the models directory.

        Returns:
            Dictionary mapping model names to their directory paths
        """
        model_dirs = {}

        if not self.models_dir.exists():
            print(f"⚠️  Warning: Models directory '{self.models_dir}' not found")
            return model_dirs

        # Look for directories matching pattern: generated_tests_*_{model_name}
        for item in self.models_dir.iterdir():
            if item.is_dir() and item.name.startswith("generated_tests_"):
                # Extract model name from directory name
                # Pattern: generated_tests_{dataset}_{model_name}
                parts = item.name.split("_")
                if len(parts) >= 3:
                    # Join all parts after "generated_tests" as model name
                    # This handles model names with underscores
                    model_name = "_".join(parts[2:])
                    model_dirs[model_name] = item
                    print(f"✓ Detected model directory: {item.name} → {model_name}")

        return model_dirs

    def load_all_models_data(self) -> None:
        """Load data from all detected model directories."""
        print("=" * 80)
        print("LOADING CROSS-MODEL DATA")
        print("=" * 80)

        model_dirs = self._detect_model_directories()

        if not model_dirs:
            print(
                "⚠️  No model directories found. Make sure you have directories matching 'generated_tests_*' pattern."
            )
            return

        # Load data for each model
        for model_name, model_dir in model_dirs.items():
            print(f"\n📦 Loading data for model: {model_name}")
            print(f"   Directory: {model_dir}")

            data_loader = DataLoader(str(model_dir), self.dataset_path)
            model_data = data_loader.load_data()

            if model_data:
                self.model_data[model_name] = model_data
                print(f"   ✓ Loaded {len(model_data)} records")
            else:
                print(f"   ⚠️  No data found for {model_name}")

        # Initialize plotting component if we have data
        if self.model_data:
            # Detect dataset type
            self.is_humanevalpack = self._detect_humanevalpack_data()
            if self.is_humanevalpack:
                print(
                    "\n📊 Detected HumanEvalPack dataset - using bug detection success criteria"
                )

            self.cross_model_plots = CrossModelPlots(self.model_data, self.config_order)
            print(f"\n✅ Successfully loaded data for {len(self.model_data)} models")
        else:
            print("\n⚠️  No data loaded for any models")

    def create_visualizations(self, output_dir: str = "cross_model_vizs") -> None:
        """Create all cross-model comparison visualizations."""
        if not self.model_data:
            print("❌ No data loaded. Call load_all_models_data() first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("\n" + "=" * 80)
        print("CREATING CROSS-MODEL VISUALIZATIONS")
        print("=" * 80)
        print(f"Output directory: {output_path}")

        if self.cross_model_plots:
            self.cross_model_plots.create_all_plots(output_path)
        else:
            print("❌ Cross-model plots component not initialized")

        print(f"\n✅ All visualizations saved to '{output_path}/' directory")

    def print_summary_stats(self) -> None:
        """Print summary statistics for all models."""
        if not self.model_data:
            print("❌ No data loaded.")
            return

        print("\n" + "=" * 80)
        print("CROSS-MODEL SUMMARY STATISTICS")
        print("=" * 80)

        # Get pricing information from models config
        pricing_info = {}
        if "models" in self.models_config:
            for model_name, config in self.models_config["models"].items():
                folder_name = config.get("folder_name", model_name)
                if "pricing" in config:
                    pricing_info[folder_name] = config["pricing"]

        for model_name in sorted(self.model_data.keys()):
            data = self.model_data[model_name]

            print(f"\n{'─' * 80}")
            print(f"MODEL: {model_name}")
            print(f"{'─' * 80}")

            # Basic stats
            total = len(data)
            success_count = sum(1 for d in data if self._is_success(d))
            success_rate = (success_count / total * 100) if total > 0 else 0

            print(f"  Total problems: {total}")
            if self.is_humanevalpack:
                print(
                    f"  Bugs detected (True Positives): {success_count} ({success_rate:.1f}%)"
                )
            else:
                print(f"  Successful tests: {success_count} ({success_rate:.1f}%)")

            # Cost stats
            costs = [d.get("total_cost", 0) for d in data if d.get("total_cost", 0) > 0]
            if costs:
                total_cost = sum(costs)
                avg_cost = sum(costs) / len(costs)
                min_cost = min(costs)
                max_cost = max(costs)
                print(f"\n  Cost Statistics:")
                print(f"    Total cost: ${total_cost:.4f}")
                print(f"    Average cost per problem: ${avg_cost:.4f}")
                print(f"    Min cost: ${min_cost:.4f}")
                print(f"    Max cost: ${max_cost:.4f}")

                # Add pricing info if available
                if model_name in pricing_info:
                    pricing = pricing_info[model_name]
                    print(
                        f"    Pricing: ${pricing.get('input_per_1M', 0)}/1M input, ${pricing.get('output_per_1M', 0)}/1M output"
                    )

            # Coverage stats (include all records for consistency)
            coverages = [d.get("coverage_percentage", 0) for d in data]
            if coverages:
                avg_coverage = sum(coverages) / len(coverages)
                # For min/max, only consider non-zero values
                non_zero_coverages = [c for c in coverages if c > 0]
                if non_zero_coverages:
                    min_coverage = min(non_zero_coverages)
                    max_coverage = max(non_zero_coverages)
                    print(f"\n  Coverage Statistics:")
                    print(f"    Average coverage: {avg_coverage:.1f}%")
                    print(f"    Min coverage: {min_coverage:.1f}%")
                    print(f"    Max coverage: {max_coverage:.1f}%")
                else:
                    print(f"\n  Coverage Statistics:")
                    print(f"    Average coverage: {avg_coverage:.1f}%")

            # Fix attempts stats
            attempts = [d.get("fix_attempts", 0) for d in data]
            if attempts:
                avg_attempts = sum(attempts) / len(attempts)
                max_attempts = max(attempts)
                print(f"\n  Fix Attempts:")
                print(f"    Average: {avg_attempts:.1f}")
                print(f"    Max: {max_attempts}")

            # Configuration breakdown
            print(f"\n  By Configuration:")
            for config in self.config_order:
                config_data = [d for d in data if d.get("config_type") == config]
                if config_data:
                    config_success = sum(1 for d in config_data if self._is_success(d))
                    config_rate = (
                        config_success / len(config_data) * 100 if config_data else 0
                    )
                    print(
                        f"    {config:15s}: {config_success:3d}/{len(config_data):3d} ({config_rate:5.1f}%)"
                    )

    def print_comparison_table(self) -> None:
        """Print a comparison table of all models."""
        if not self.model_data:
            print("❌ No data loaded.")
            return

        print("\n" + "=" * 80)
        print("CROSS-MODEL COMPARISON TABLE")
        print("=" * 80)

        # Calculate metrics for each model
        metrics = []
        for model_name in sorted(self.model_data.keys()):
            data = self.model_data[model_name]

            total = len(data)
            success_count = sum(1 for d in data if self._is_success(d))
            success_rate = (success_count / total * 100) if total > 0 else 0

            costs = [d.get("total_cost", 0) for d in data if d.get("total_cost", 0) > 0]
            avg_cost = (sum(costs) / len(costs)) if costs else 0

            # Include all records for consistency with single-model analysis
            coverages = [d.get("coverage_percentage", 0) for d in data]
            avg_coverage = (sum(coverages) / len(coverages)) if coverages else 0

            attempts = [d.get("fix_attempts", 0) for d in data]
            avg_attempts = (sum(attempts) / len(attempts)) if attempts else 0

            efficiency = (success_rate / avg_cost) if avg_cost > 0 else 0

            metrics.append(
                {
                    "model": model_name,
                    "total": total,
                    "success_rate": success_rate,
                    "avg_cost": avg_cost,
                    "avg_coverage": avg_coverage,
                    "avg_attempts": avg_attempts,
                    "efficiency": efficiency,
                }
            )

        # Print table header
        print(
            f"\n{'Model':<25} {'Total':<8} {'Success%':<10} {'Avg Cost':<12} {'Avg Cov%':<10} {'Avg Attempts':<13} {'Efficiency':<12}"
        )
        print("─" * 120)

        # Print metrics for each model
        for m in metrics:
            print(
                f"{m['model']:<25} {m['total']:<8} {m['success_rate']:>8.1f}% "
                f"${m['avg_cost']:>9.4f}  {m['avg_coverage']:>8.1f}% "
                f"{m['avg_attempts']:>11.1f}  {m['efficiency']:>10.1f}"
            )

        print("─" * 120)
        print("\nNote: Efficiency = Success Rate / Average Cost (higher is better)")

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary information about loaded data."""
        return {
            "total_models": len(self.model_data),
            "models": list(self.model_data.keys()),
            "total_records": sum(len(data) for data in self.model_data.values()),
        }


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-Model Test Results Analysis Tool - Compare test results across multiple models"
    )
    parser.add_argument(
        "--models-dir",
        default="data",
        help="Directory containing model result directories (default: data)",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to the dataset file (optional - will load from HuggingFace if not provided)",
    )
    parser.add_argument(
        "--models-config",
        default="models_config.json",
        help="Path to models configuration JSON (default: models_config.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="cross_model_vizs",
        help="Output directory for visualization files (default: cross_model_vizs)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the cross-model analysis tool."""
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print("🔍 Cross-Model Test Results Analysis Tool")
    print("=" * 50)
    print(f"📁 Models directory: {args.models_dir}")
    if args.dataset_path:
        print(f"📊 Dataset path: {args.dataset_path}")
    else:
        print(f"📊 Dataset source: HuggingFace (openai/openai_humaneval)")
    print(f"⚙️  Models config: {args.models_config}")
    print(f"📈 Output directory: {args.output_dir}")

    # Initialize analyzer
    dataset_path = args.dataset_path if args.dataset_path else "dataset/HumanEval.jsonl"
    analyzer = CrossModelAnalyzer(
        models_dir=args.models_dir,
        dataset_path=dataset_path,
        models_config_path=args.models_config,
    )

    # Load data from all models
    analyzer.load_all_models_data()

    # Print data summary
    summary = analyzer.get_data_summary()
    print(f"\n📊 Data Summary:")
    print(f"  Total models: {summary['total_models']}")
    print(f"  Models: {', '.join(summary['models'])}")
    print(f"  Total records: {summary['total_records']}")

    if summary["total_models"] == 0:
        print(
            "\n❌ No model data found. Make sure you have directories with .stats.json files."
        )
        return

    # Create visualizations
    analyzer.create_visualizations(args.output_dir)

    # Print analysis reports
    analyzer.print_summary_stats()
    analyzer.print_comparison_table()

    print(
        f"\n🎉 Cross-model analysis complete! Check the '{args.output_dir}' directory for graphs."
    )


if __name__ == "__main__":
    main()

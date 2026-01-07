#!/usr/bin/env python3
"""
プロンプトエンジニアリング手法の性能比較グラフ作成スクリプト

4種類のプロンプトエンジニアリング手法（Basic, Docstring, AST, Docstring-AST）の
Success RateとCoverageを比較するグラフを作成します。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Hiragino Sans",
    "Yu Gothic",
    "Meirio",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class PromptEngineeringAnalyzer:
    """プロンプトエンジニアリング手法の性能分析クラス"""

    def __init__(self, data_dir: str):
        """
        初期化

        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
        self.data = []

        # プロンプト手法の定義と表示名
        self.prompt_types = {
            "basic": "Basic",
            "docstring": "Docstring",
            "ast": "AST",
            "docstring_ast": "Docstring-AST",
        }

    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        ファイル名から設定情報を解析

        Args:
            filename: ファイル名

        Returns:
            設定情報の辞書
        """
        # .stats.json拡張子を除去
        base_name = filename.replace(".stats.json", "")

        # 問題IDの抽出
        match = re.search(r"test_humaneval_(\d+)", base_name)
        if not match:
            return None

        problem_id = int(match.group(1))

        # 設定フラグの確認
        has_docstring = "_docstring" in base_name
        has_ast = "_ast" in base_name

        # 成功ステータスの確認
        is_success = base_name.endswith("_success")

        # 設定タイプの決定
        if has_docstring and has_ast:
            config_type = "docstring_ast"
        elif has_docstring:
            config_type = "docstring"
        elif has_ast:
            config_type = "ast"
        else:
            config_type = "basic"

        return {
            "problem_id": problem_id,
            "config_type": config_type,
            "success": is_success,
            "filename": filename,
        }

    def load_data(self) -> None:
        """データの読み込み"""
        print(f"Loading data from {self.data_dir}...")

        stats_files = list(self.data_dir.glob("*.stats.json"))
        print(f"Found {len(stats_files)} stats files")

        for stats_file in stats_files:
            # ファイル名から設定情報を解析
            file_config = self.parse_filename(stats_file.name)
            if not file_config:
                print(f"Warning: Could not parse filename {stats_file.name}")
                continue

            # statsデータを読み込み
            try:
                with open(stats_file, "r") as f:
                    stats_data = json.load(f)

                # ファイル名情報とstatsデータを結合
                combined_data = {**file_config, **stats_data}

                # code_coverage_percentフィールドが欠けている場合は0.0を設定
                if "code_coverage_percent" not in combined_data:
                    combined_data["code_coverage_percent"] = 0.0

                self.data.append(combined_data)

            except Exception as e:
                print(f"Error loading {stats_file}: {e}")

        print(f"Successfully loaded {len(self.data)} records")

        # 各設定タイプの数を表示
        config_counts = {}
        for record in self.data:
            config = record["config_type"]
            config_counts[config] = config_counts.get(config, 0) + 1

        print("\nConfiguration counts:")
        for config, count in sorted(config_counts.items()):
            print(f"  {config}: {count}")

    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        各プロンプト手法の評価指標を計算

        Returns:
            各プロンプト手法の指標を含む辞書
        """
        metrics = {}

        for config_type in self.prompt_types.keys():
            # 該当する設定タイプのデータをフィルタリング
            config_data = [d for d in self.data if d["config_type"] == config_type]

            if not config_data:
                print(f"Warning: No data found for {config_type}")
                continue

            # Success Rate: evaluation_successがTrueの割合
            total_count = len(config_data)
            success_count = sum(
                1 for d in config_data if d.get("evaluation_success", False)
            )
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0

            # Coverage: code_coverage_percentの平均
            coverage_values = [d.get("code_coverage_percent", 0) for d in config_data]
            avg_coverage = np.mean(coverage_values) if coverage_values else 0

            metrics[config_type] = {
                "success_rate": success_rate,
                "coverage": avg_coverage,
                "total_count": total_count,
                "success_count": success_count,
            }

            print(f"\n{self.prompt_types[config_type]}:")
            print(
                f"  Success Rate: {success_rate:.2f}% ({success_count}/{total_count})"
            )
            print(f"  Average Coverage: {avg_coverage:.2f}%")

        return metrics

    def create_comparison_plots(
        self, metrics: Dict[str, Dict[str, float]], output_dir: str = "."
    ) -> None:
        """
        比較グラフを作成

        Args:
            metrics: 各プロンプト手法の指標
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # データの準備
        configs = list(self.prompt_types.keys())
        config_labels = [self.prompt_types[c] for c in configs]
        success_rates = [metrics[c]["success_rate"] for c in configs]
        coverages = [metrics[c]["coverage"] for c in configs]

        # カラーパレット
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        # 1. Success Rate比較グラフ
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            config_labels, success_rates, color=colors, alpha=0.8, edgecolor="black"
        )
        ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Prompt Engineering Method", fontsize=12, fontweight="bold")
        ax.set_title(
            "Success Rate Comparison by Prompt Engineering Method",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # 値をバーの上に表示
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            output_path / "success_rate_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(f"\n✅ Saved: {output_path / 'success_rate_comparison.png'}")
        plt.close()

        # 2. Coverage比較グラフ
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            config_labels, coverages, color=colors, alpha=0.8, edgecolor="black"
        )
        ax.set_ylabel("Average Code Coverage (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Prompt Engineering Method", fontsize=12, fontweight="bold")
        ax.set_title(
            "Code Coverage Comparison by Prompt Engineering Method",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # 値をバーの上に表示
        for bar, cov in zip(bars, coverages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{cov:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            output_path / "coverage_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(f"✅ Saved: {output_path / 'coverage_comparison.png'}")
        plt.close()

        # 3. 統合比較グラフ（サイドバイサイド）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Success Rate
        bars1 = ax1.bar(
            config_labels, success_rates, color=colors, alpha=0.8, edgecolor="black"
        )
        ax1.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Prompt Engineering Method", fontsize=12, fontweight="bold")
        ax1.set_title("Success Rate", fontsize=13, fontweight="bold")
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Coverage
        bars2 = ax2.bar(
            config_labels, coverages, color=colors, alpha=0.8, edgecolor="black"
        )
        ax2.set_ylabel("Average Code Coverage (%)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Prompt Engineering Method", fontsize=12, fontweight="bold")
        ax2.set_title("Code Coverage", fontsize=13, fontweight="bold")
        ax2.set_ylim(0, 100)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        for bar, cov in zip(bars2, coverages):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{cov:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        fig.suptitle(
            "Prompt Engineering Methods Comparison: Success Rate and Coverage",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(
            output_path / "combined_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(f"✅ Saved: {output_path / 'combined_comparison.png'}")
        plt.close()

        # 4. レーダーチャート
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        angles = np.linspace(0, 2 * np.pi, len(config_labels), endpoint=False).tolist()
        angles += angles[:1]  # 閉じるために最初の角度を追加

        # Success Rateをプロット
        values_sr = success_rates + [success_rates[0]]
        ax.plot(
            angles, values_sr, "o-", linewidth=2, label="Success Rate", color="#3498db"
        )
        ax.fill(angles, values_sr, alpha=0.25, color="#3498db")

        # Coverageをプロット
        values_cov = coverages + [coverages[0]]
        ax.plot(
            angles, values_cov, "o-", linewidth=2, label="Coverage", color="#e74c3c"
        )
        ax.fill(angles, values_cov, alpha=0.25, color="#e74c3c")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(config_labels, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9)
        ax.set_title(
            "Radar Chart: Success Rate vs Coverage\nby Prompt Engineering Method",
            fontsize=14,
            fontweight="bold",
            pad=30,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "radar_comparison.png", dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {output_path / 'radar_comparison.png'}")
        plt.close()

        # 5. 散布図（Success Rate vs Coverage）
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, (config, label, color) in enumerate(zip(configs, config_labels, colors)):
            sr = success_rates[i]
            cov = coverages[i]
            ax.scatter(
                sr, cov, s=500, alpha=0.7, color=color, edgecolors="black", linewidth=2
            )
            ax.text(
                sr, cov, label, ha="center", va="center", fontsize=10, fontweight="bold"
            )

        ax.set_xlabel("Success Rate (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Code Coverage (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Success Rate vs Code Coverage\nby Prompt Engineering Method",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, linestyle="--")

        # 理想的な領域を示す参照線
        ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80% Coverage")
        ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5, label="80% Success")
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(
            output_path / "scatter_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(f"✅ Saved: {output_path / 'scatter_comparison.png'}")
        plt.close()


def main():
    """メイン関数"""
    # 全てのClaudeモデルのデータディレクトリ
    base_data_dir = "/Users/soso/pj-aidev-PromptEng2025/data"
    output_base_dir = "/Users/soso/pj-aidev-PromptEng2025/prompt_comparison_results"

    # 分析対象のモデルリスト
    models = [
        "generated_tests_claude-3-5-haiku",
        "generated_tests_claude-3-haiku",
        "generated_tests_claude-4-5-sonnet",
        "generated_tests_claude-4-sonnet",
        "generated_tests_claude-haiku-4-5",
        "generated_tests_claude-opus-4-1",
    ]

    print("=" * 80)
    print("プロンプトエンジニアリング手法 性能比較分析（全Claudeモデル）")
    print("=" * 80)
    print(f"ベースデータディレクトリ: {base_data_dir}")
    print(f"出力ベースディレクトリ: {output_base_dir}")
    print(f"分析対象モデル数: {len(models)}")
    print("=" * 80)

    # 全モデル統合用のアナライザー
    combined_analyzer = PromptEngineeringAnalyzer("")
    combined_analyzer.data = []  # 空のデータリストで初期化

    # 除外モデルを除いた統合用のアナライザー
    excluded_models = [
        "generated_tests_claude-3-haiku",
        "generated_tests_claude-opus-4-1",
    ]
    filtered_analyzer = PromptEngineeringAnalyzer("")
    filtered_analyzer.data = []  # 空のデータリストで初期化

    # 各モデルに対して分析を実行
    for i, model in enumerate(models, 1):
        data_dir = Path(base_data_dir) / model
        output_dir = Path(output_base_dir) / model

        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(models)}] モデル: {model}")
        print("=" * 80)
        print(f"データディレクトリ: {data_dir}")
        print(f"出力ディレクトリ: {output_dir}")

        # ディレクトリの存在確認
        if not data_dir.exists():
            print(f"⚠️  警告: データディレクトリが見つかりません: {data_dir}")
            print("スキップします...")
            continue

        try:
            # アナライザーの初期化とデータ読み込み
            analyzer = PromptEngineeringAnalyzer(str(data_dir))
            analyzer.load_data()

            # 統合アナライザーにデータを追加
            combined_analyzer.data.extend(analyzer.data)

            # 除外リストにないモデルのデータをフィルター済みアナライザーに追加
            if model not in excluded_models:
                filtered_analyzer.data.extend(analyzer.data)

            print("\n" + "-" * 80)
            print("評価指標の計算")
            print("-" * 80)

            # メトリクスの計算
            metrics = analyzer.calculate_metrics()

            print("\n" + "-" * 80)
            print("グラフの作成")
            print("-" * 80)

            # グラフの作成
            analyzer.create_comparison_plots(metrics, str(output_dir))

            print(f"\n✅ {model} の分析が完了しました！")

        except Exception as e:
            print(f"\n❌ エラー: {model} の分析中にエラーが発生しました")
            print(f"   {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # 全モデル統合結果の分析
    print("\n" + "=" * 80)
    print("全モデル統合結果の分析")
    print("=" * 80)

    if combined_analyzer.data:
        output_dir_combined = Path(output_base_dir) / "all_models_combined"
        print(f"統合データ総数: {len(combined_analyzer.data)} records")
        print(f"出力ディレクトリ: {output_dir_combined}")

        try:
            # 各設定タイプの数を表示
            config_counts = {}
            for record in combined_analyzer.data:
                config = record["config_type"]
                config_counts[config] = config_counts.get(config, 0) + 1

            print("\nCombined configuration counts:")
            for config, count in sorted(config_counts.items()):
                print(f"  {config}: {count}")

            print("\n" + "-" * 80)
            print("統合評価指標の計算")
            print("-" * 80)

            # 統合メトリクスの計算
            combined_metrics = combined_analyzer.calculate_metrics()

            print("\n" + "-" * 80)
            print("統合グラフの作成")
            print("-" * 80)

            # 統合グラフの作成
            combined_analyzer.create_comparison_plots(
                combined_metrics, str(output_dir_combined)
            )

            print(f"\n✅ 全モデル統合結果の分析が完了しました！")

        except Exception as e:
            print(f"\n❌ エラー: 統合結果の分析中にエラーが発生しました")
            print(f"   {str(e)}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️  警告: 統合データがありません")

    # フィルター済みモデル統合結果の分析（claude-3-haiku と opus-4-1 を除外）
    print("\n" + "=" * 80)
    print("フィルター済みモデル統合結果の分析")
    print(f"(除外: {', '.join(excluded_models)})")
    print("=" * 80)

    if filtered_analyzer.data:
        output_dir_filtered = Path(output_base_dir) / "filtered_models_combined"
        print(f"フィルター済み統合データ総数: {len(filtered_analyzer.data)} records")
        print(f"出力ディレクトリ: {output_dir_filtered}")

        try:
            # 各設定タイプの数を表示
            config_counts = {}
            for record in filtered_analyzer.data:
                config = record["config_type"]
                config_counts[config] = config_counts.get(config, 0) + 1

            print("\nFiltered configuration counts:")
            for config, count in sorted(config_counts.items()):
                print(f"  {config}: {count}")

            print("\n" + "-" * 80)
            print("フィルター済み評価指標の計算")
            print("-" * 80)

            # フィルター済みメトリクスの計算
            filtered_metrics = filtered_analyzer.calculate_metrics()

            print("\n" + "-" * 80)
            print("フィルター済みグラフの作成")
            print("-" * 80)

            # フィルター済みグラフの作成
            filtered_analyzer.create_comparison_plots(
                filtered_metrics, str(output_dir_filtered)
            )

            print(f"\n✅ フィルター済みモデル統合結果の分析が完了しました！")

        except Exception as e:
            print(f"\n❌ エラー: フィルター済み統合結果の分析中にエラーが発生しました")
            print(f"   {str(e)}")
            import traceback

            traceback.print_exc()
    else:
        print("⚠️  警告: フィルター済み統合データがありません")

    print("\n" + "=" * 80)
    print("全ての分析が完了！")
    print("=" * 80)
    print(f"\nすべてのグラフが '{output_base_dir}/' に保存されました。")
    print("\n各ディレクトリに作成されたグラフ:")
    print("  - 個別モデルディレクトリ (6個)")
    print("  - all_models_combined (全6モデル統合)")
    print("  - filtered_models_combined (claude-3-haiku, opus-4-1除外)")
    print("\n各ディレクトリ内のグラフ:")
    print("  1. success_rate_comparison.png - Success Rate比較")
    print("  2. coverage_comparison.png - Coverage比較")
    print("  3. combined_comparison.png - 統合比較（並列）")
    print("  4. radar_comparison.png - レーダーチャート")
    print("  5. scatter_comparison.png - 散布図（Success Rate vs Coverage）")


if __name__ == "__main__":
    main()

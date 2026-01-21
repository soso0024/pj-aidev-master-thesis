#!/usr/bin/env python3
"""
Display Problem Classification Details

Shows how HumanEval problems are classified into complexity levels
based on cognitive complexity scores.
Also visualizes bug type distribution from HumanEvalPack dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports from analysis module
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from analysis.problem_classifier import ProblemClassifier

# Set Japanese font for matplotlib
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Import datasets library for HumanEvalPack
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Bug type visualization will be skipped.")


def load_bug_type_distribution():
    """Load bug type distribution from HumanEvalPack dataset.
    
    Returns:
        dict: Bug type counts, or None if dataset not available
    """
    if not DATASETS_AVAILABLE:
        return None
    
    try:
        print("\nLoading HumanEvalPack dataset from Hugging Face...")
        dataset = load_dataset("bigcode/humanevalpack", "python", split="test")
        
        # Count bug types
        bug_type_counts = {}
        for item in dataset:
            bug_type = item.get("bug_type", "unknown")
            bug_type_counts[bug_type] = bug_type_counts.get(bug_type, 0) + 1
        
        print(f"✅ Loaded {len(dataset)} problems from HumanEvalPack dataset")
        return bug_type_counts
    
    except Exception as e:
        print(f"❌ Failed to load HumanEvalPack dataset: {e}")
        return None


def visualize_bug_types(bug_type_counts, output_dir="problem_classification"):
    """Create visualization for bug type distribution.
    
    Args:
        bug_type_counts: Dictionary of bug type counts
        output_dir: Directory to save the visualization
    """
    if not bug_type_counts:
        return
    
    print("\n" + "=" * 80)
    print("BUG TYPE DISTRIBUTION (HumanEvalPack)")
    print("=" * 80)
    
    # Sort bug types by count (descending)
    sorted_bug_types = sorted(bug_type_counts.items(), key=lambda x: x[1], reverse=True)
    bug_types = [bt[0] for bt in sorted_bug_types]
    counts = [bt[1] for bt in sorted_bug_types]
    
    # Map English bug types to Japanese
    bug_type_translation = {
        'value misuse': '値の誤用',
        'missing logic': 'ロジック不足',
        'excess logic': '余分なロジック',
        'operator misuse': '演算子の誤用',
        'variable misuse': '変数の誤用',
        'function misuse': '関数の誤用'
    }
    bug_types_jp = [bug_type_translation.get(bt, bt) for bt in bug_types]
    
    total_problems = sum(counts)
    
    print(f"\nTotal Problems: {total_problems}")
    print(f"\nDistribution by Bug Type:")
    for bug_type, count in sorted_bug_types:
        percentage = (count / total_problems) * 100
        print(f"  {bug_type:<20}: {count:3d} problems ({percentage:5.1f}%)")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define color palette for bug types
    colors = ['#FF4B00', '#005AFF', '#03AF7A', '#4DC4FF', '#F6AA00', '#FFF100']
    bar_colors = [colors[i % len(colors)] for i in range(len(bug_types))]
    
    bars = ax.bar(range(len(bug_types)), counts, color=bar_colors, 
                   alpha=0.85, edgecolor='black', linewidth=2)
    
    # Set x-axis labels (in Japanese)
    ax.set_xticks(range(len(bug_types)))
    ax.set_xticklabels(bug_types_jp, rotation=45, ha='right', fontsize=12, fontweight='bold')
    
    # Set y-axis label (in Japanese)
    ax.set_ylabel('問題数', fontsize=14, fontweight='bold')
    
    # No title (as requested)
    ax.set_ylim(0, 50)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (counts[i] / total_problems) * 100
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = f"{output_dir}/bug_type_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def main():
    """Display classification details and create visualizations."""
    print("=" * 80)
    print("PROBLEM CLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    # Initialize classifier
    classifier = ProblemClassifier(
        dataset_path="dataset/HumanEval.jsonl",
        use_adaptive_thresholds=True
    )
    
    # Load classifications
    print("\nLoading problem classifications...")
    classifier.load_problem_classifications()
    
    classifications = classifier.get_all_classifications()
    
    if not classifications:
        print("No classifications found.")
        return
    
    # Extract data
    problem_ids = sorted(classifications.keys())
    complexity_scores = [classifications[pid]["complexity_score"] for pid in problem_ids]
    complexity_levels = [classifications[pid]["complexity_level"] for pid in problem_ids]
    
    # Count by level
    level_counts = {
        "simple": sum(1 for level in complexity_levels if level == "simple"),
        "medium": sum(1 for level in complexity_levels if level == "medium"),
        "complex": sum(1 for level in complexity_levels if level == "complex"),
        "error": sum(1 for level in complexity_levels if level == "error"),
    }
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal Problems: {len(classifications)}")
    print(f"\nDistribution by Complexity Level:")
    for level, count in level_counts.items():
        percentage = (count / len(classifications)) * 100 if classifications else 0
        print(f"  {level.capitalize():<10}: {count:3d} problems ({percentage:5.1f}%)")
    
    # Print threshold information
    if classifier.complexity_thresholds:
        low_threshold, high_threshold = classifier.complexity_thresholds
        print(f"\nAdaptive Thresholds:")
        print(f"  Simple/Medium boundary:  {low_threshold:.2f}")
        print(f"  Medium/Complex boundary: {high_threshold:.2f}")
    
    # Statistics
    print(f"\nCognitive Complexity Statistics:")
    print(f"  Mean:   {np.mean(complexity_scores):.2f}")
    print(f"  Median: {np.median(complexity_scores):.2f}")
    print(f"  Std:    {np.std(complexity_scores):.2f}")
    print(f"  Min:    {np.min(complexity_scores):.2f}")
    print(f"  Max:    {np.max(complexity_scores):.2f}")
    
    # Print sample problems from each category
    print("\n" + "=" * 80)
    print("SAMPLE PROBLEMS BY COMPLEXITY LEVEL")
    print("=" * 80)
    
    for level in ["simple", "medium", "complex"]:
        print(f"\n{level.upper()} Problems (showing up to 5):")
        sample_problems = [pid for pid in problem_ids 
                          if classifications[pid]["complexity_level"] == level][:5]
        for pid in sample_problems:
            score = classifications[pid]["complexity_score"]
            print(f"  Problem {pid:3d}: Cognitive Complexity = {score:.1f}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Define color scheme
    # Simple: Green (#03AF7A), Medium: Blue (#005AFF), Complex: Orange/Red (#FF4B00)
    COLOR_SIMPLE = '#03AF7A'
    COLOR_MEDIUM = '#005AFF'
    COLOR_COMPLEX = '#FF4B00'
    
    # ========================================================================
    # VISUALIZATION 1: Distribution of Cognitive Complexity Scores
    # ========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Separate complexity scores by level
    simple_scores = [classifications[pid]['complexity_score'] 
                    for pid in problem_ids if classifications[pid]['complexity_level'] == 'simple']
    medium_scores = [classifications[pid]['complexity_score'] 
                    for pid in problem_ids if classifications[pid]['complexity_level'] == 'medium']
    complex_scores = [classifications[pid]['complexity_score'] 
                     for pid in problem_ids if classifications[pid]['complexity_level'] == 'complex']
    
    # Define bins for histogram - 整数スコアに合わせて1間隔のビンを使用
    # 離散値（整数）のため、各整数の中心にバーが来るようにビンを設定（例：0の場合は-0.5〜0.5）
    max_score = int(max(complexity_scores))
    bins = np.arange(-0.5, max_score + 1.5, 1)
    
    # Plot histograms for each level with different colors
    ax1.hist(simple_scores, bins=bins, color=COLOR_SIMPLE, 
             alpha=0.75, label='Simple')
    ax1.hist(medium_scores, bins=bins, color=COLOR_MEDIUM, 
             alpha=0.75, label='Medium')
    ax1.hist(complex_scores, bins=bins, color=COLOR_COMPLEX, 
             alpha=0.75, label='Complex')
    
    ax1.set_xlabel('認知的複雑度スコア', fontsize=14, fontweight='bold')
    ax1.set_ylabel('問題数', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, framealpha=0.9, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)  # X軸を0から開始（0のバーは半分切れるが、マイナス表示を避けるため）

    # 閾値の線を追加
    if classifier.complexity_thresholds:
        low_threshold, high_threshold = classifier.complexity_thresholds
        
        # 視覚的な境界線（整数ビンの間）に設定
        # 1.32 -> 1.5 (1と2の間), 5.33 -> 5.5 (5と6の間)
        visual_low = 1.5
        visual_high = 5.5
        
        # Simple/Medium 境界線
        ax1.axvline(x=visual_low, color='#03AF7A', linestyle='--', linewidth=4, alpha=0.8)
        
        # Medium/Complex 境界線
        ax1.axvline(x=visual_high, color='#FF4B00', linestyle='--', linewidth=4, alpha=0.8)

        # X軸の目盛りを調整（閾値を含める、5を除外）
        # 基本の整数目盛り（0, 5, 10...）から5を除外
        base_ticks = [t for t in range(0, max_score + 2, 5) if t != 5]
        
        # 閾値を追加してソート
        custom_ticks = sorted(list(set(base_ticks + [visual_low, visual_high])))
        ax1.set_xticks(custom_ticks)
        
        # ラベルのフォーマット（整数は整数表示、小数は小数表示）
        # 視覚的な位置(1.5, 5.5)には実際の閾値(1.3, 5.3)を表示する
        labels = []
        for tick in custom_ticks:
            if tick == visual_low:
                labels.append(f'{low_threshold:.1f}')
            elif tick == visual_high:
                labels.append(f'{high_threshold:.1f}')
            else:
                labels.append(f'{int(tick)}')
        ax1.set_xticklabels(labels, rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('problem_classification/complexity_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: problem_classification/complexity_distribution.png")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 2: Problems by Complexity Level (Bar Chart)
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    levels = ['Simple', 'Medium', 'Complex']
    counts = [level_counts['simple'], level_counts['medium'], level_counts['complex']]
    colors = [COLOR_SIMPLE, COLOR_MEDIUM, COLOR_COMPLEX]
    bars = ax2.bar(levels, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax2.set_ylabel('問題数', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('problem_classification/problems_by_level.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: problem_classification/problems_by_level.png")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 3: Bug Type Distribution from HumanEvalPack
    # ========================================================================
    bug_type_counts = load_bug_type_distribution()
    if bug_type_counts:
        visualize_bug_types(bug_type_counts)
    else:
        print("\n⚠️  Skipping bug type visualization (dataset not available)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


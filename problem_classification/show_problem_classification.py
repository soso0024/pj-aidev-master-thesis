#!/usr/bin/env python3
"""
Display Problem Classification Details

Shows how HumanEval problems are classified into complexity levels
based on cognitive complexity scores.
"""

import matplotlib.pyplot as plt
import numpy as np
from analysis.problem_classifier import ProblemClassifier


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
    
    # Define bins for histogram
    bins = 20
    bin_range = (0, max(complexity_scores) + 1)
    
    # Plot histograms for each level with different colors
    ax1.hist(simple_scores, bins=bins, range=bin_range, color=COLOR_SIMPLE, 
             alpha=0.75, edgecolor='black', linewidth=1.2, label='Simple')
    ax1.hist(medium_scores, bins=bins, range=bin_range, color=COLOR_MEDIUM, 
             alpha=0.75, edgecolor='black', linewidth=1.2, label='Medium')
    ax1.hist(complex_scores, bins=bins, range=bin_range, color=COLOR_COMPLEX, 
             alpha=0.75, edgecolor='black', linewidth=1.2, label='Complex')
    
    # Add threshold lines with different line styles (both black)
    if classifier.complexity_thresholds:
        low_t, high_t = classifier.complexity_thresholds
        ax1.axvline(low_t, color='black', linestyle='--', linewidth=2.5, 
                   label=f'Simple/Medium: {low_t:.1f}')
        ax1.axvline(high_t, color='black', linestyle=':', linewidth=2.5, 
                   label=f'Medium/Complex: {high_t:.1f}')
    
    ax1.set_xlabel('Cognitive Complexity Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Problems', fontsize=14, fontweight='bold')
    ax1.set_title('Distribution of Cognitive Complexity Scores', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, framealpha=0.9, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('complexity_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: complexity_distribution.png")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 2: Problems by Complexity Level (Bar Chart)
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    levels = ['Simple', 'Medium', 'Complex']
    counts = [level_counts['simple'], level_counts['medium'], level_counts['complex']]
    colors = [COLOR_SIMPLE, COLOR_MEDIUM, COLOR_COMPLEX]
    bars = ax2.bar(levels, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Problems', fontsize=14, fontweight='bold')
    ax2.set_title('Problems by Complexity Level', fontsize=16, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('problems_by_level.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: problems_by_level.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


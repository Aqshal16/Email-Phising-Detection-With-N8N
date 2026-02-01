"""
Model Comparison Script - XGBoost with/without OSINT
Comparison study sesuai paper An et al. (2025)

Compare:
1. XGBoost Text-Only (baseline)
2. XGBoost + OSINT (17 features from paper)

Generate:
- Accuracy comparison table
- Confusion matrix comparison
- Feature importance analysis
- Performance metrics plot
- Improvement analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# ========================================
# KONFIGURASI
# ========================================
RESULTS_FILE = "results/training_results_hybrid.json"
OUTPUT_DIR = "results/comparison"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# LOAD RESULTS
# ========================================

def load_results():
    """Load training results from JSON."""
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Error: Results file not found: {RESULTS_FILE}")
        print(f"\n💡 Please run training first:")
        print(f"   1. python train_model_hybrid.py (MODE='TEXT_ONLY')")
        print(f"   2. python train_model_hybrid.py (MODE='OSINT_ENHANCED')")
        return None
    
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    return results

# ========================================
# COMPARISON TABLE
# ========================================

def generate_comparison_table(results):
    """Generate comparison table similar to paper Table 4."""
    
    print("\n" + "="*80)
    print("  📊 MODEL COMPARISON - XGBoost Text-Only vs OSINT-Enhanced")
    print("="*80)
    
    # Paper target results (Table 4)
    paper_targets = {
        'TEXT_ONLY': {
            'accuracy': 95.39,
            'f1_score': 95.54,
            'recall': 92.59,
            'precision': 98.68
        },
        'OSINT_ENHANCED': {
            'accuracy': 96.71,
            'f1_score': 96.73,
            'recall': 96.10,
            'precision': 97.37
        }
    }
    
    # Create comparison dataframe
    comparison_data = []
    
    for mode in ['TEXT_ONLY', 'OSINT_ENHANCED']:
        if mode in results:
            row = {
                'Model': 'XGBoost ' + ('Text-Only' if mode == 'TEXT_ONLY' else '+ OSINT'),
                'Accuracy (%)': results[mode]['accuracy'] * 100,
                'F1 Score (%)': results[mode]['f1_score'] * 100,
                'Recall (%)': results[mode]['recall'] * 100,
                'Precision (%)': results[mode]['precision'] * 100,
                'Features': results[mode]['n_features'],
                'Paper Target Acc (%)': paper_targets[mode]['accuracy']
            }
            comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display table
    print(f"\n📋 Performance Comparison (70:30 Train-Test Split):")
    print("─"*80)
    print(df_comparison.to_string(index=False))
    print("─"*80)
    
    # Calculate improvement
    if 'TEXT_ONLY' in results and 'OSINT_ENHANCED' in results:
        acc_improvement = (results['OSINT_ENHANCED']['accuracy'] - results['TEXT_ONLY']['accuracy']) * 100
        f1_improvement = (results['OSINT_ENHANCED']['f1_score'] - results['TEXT_ONLY']['f1_score']) * 100
        
        print(f"\n💡 OSINT Enhancement Impact:")
        print(f"   Accuracy improvement: +{acc_improvement:.2f}% (Paper: +1.32%)")
        print(f"   F1 Score improvement: +{f1_improvement:.2f}% (Paper: +1.19%)")
        
        # Comparison with paper
        paper_acc_improvement = 96.71 - 95.39
        print(f"\n📊 Comparison with Paper (An et al. 2025):")
        print(f"   Our accuracy improvement: {acc_improvement:.2f}%")
        print(f"   Paper accuracy improvement: {paper_acc_improvement:.2f}%")
        print(f"   Match: {'✅ Close!' if abs(acc_improvement - paper_acc_improvement) < 1 else '⚠️ Different dataset size'}")
    
    # Save table
    table_file = os.path.join(OUTPUT_DIR, "comparison_table.csv")
    df_comparison.to_csv(table_file, index=False)
    print(f"\n💾 Table saved: {table_file}")
    
    return df_comparison

# ========================================
# CONFUSION MATRIX COMPARISON
# ========================================

def plot_confusion_matrices(results):
    """Plot confusion matrices side by side (similar to paper Table 6)."""
    
    if 'TEXT_ONLY' not in results or 'OSINT_ENHANCED' not in results:
        print("⚠️  Need both TEXT_ONLY and OSINT_ENHANCED results for comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    modes = ['TEXT_ONLY', 'OSINT_ENHANCED']
    titles = ['XGBoost Text-Only', 'XGBoost + OSINT']
    
    for idx, (mode, title) in enumerate(zip(modes, titles)):
        cm = np.array(results[mode]['confusion_matrix'])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Safe', 'Phishing'],
                    yticklabels=['Safe', 'Phishing'],
                    ax=axes[idx], cbar=False)
        
        axes[idx].set_title(f'{title}\nAccuracy: {results[mode]["accuracy"]*100:.2f}%', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        # Add metrics text
        tn, fp, fn, tp = cm.ravel()
        metrics_text = f'TP={tp}, TN={tn}\nFP={fp}, FN={fn}'
        axes[idx].text(0.5, -0.15, metrics_text, ha='center', 
                      transform=axes[idx].transAxes, fontsize=9)
    
    plt.tight_layout()
    cm_file = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrices saved: {cm_file}")
    plt.close()

# ========================================
# PERFORMANCE METRICS PLOT
# ========================================

def plot_metrics_comparison(results):
    """Plot bar chart comparing metrics (similar to paper visualization)."""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Prepare data
    text_only_scores = []
    osint_enhanced_scores = []
    
    for metric in metrics:
        if 'TEXT_ONLY' in results:
            text_only_scores.append(results['TEXT_ONLY'][metric] * 100)
        else:
            text_only_scores.append(0)
        
        if 'OSINT_ENHANCED' in results:
            osint_enhanced_scores.append(results['OSINT_ENHANCED'][metric] * 100)
        else:
            osint_enhanced_scores.append(0)
    
    # Plot
    x = np.arange(len(metric_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, text_only_scores, width, label='Text-Only', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, osint_enhanced_scores, width, label='OSINT-Enhanced', 
                   color='#2ecc71', alpha=0.8)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Styling
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('XGBoost Performance Comparison: Text-Only vs OSINT-Enhanced\n' + 
                 'Based on An et al. (2025) Methodology', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([90, 100])
    
    plt.tight_layout()
    metrics_file = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    print(f"💾 Metrics comparison saved: {metrics_file}")
    plt.close()

# ========================================
# IMPROVEMENT ANALYSIS
# ========================================

def plot_improvement_analysis(results):
    """Plot improvement from Text-Only to OSINT-Enhanced."""
    
    if 'TEXT_ONLY' not in results or 'OSINT_ENHANCED' not in results:
        print("⚠️  Need both results for improvement analysis")
        return
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    improvements = []
    for metric in metrics:
        improvement = (results['OSINT_ENHANCED'][metric] - results['TEXT_ONLY'][metric]) * 100
        improvements.append(improvement)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.barh(metric_labels, improvements, color=colors, alpha=0.7)
    
    # Add value labels
    for idx, (bar, imp) in enumerate(zip(bars, improvements)):
        ax.text(imp + 0.05 if imp > 0 else imp - 0.05, idx, 
               f'{imp:+.2f}%', va='center', 
               ha='left' if imp > 0 else 'right', fontsize=10)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement with OSINT Features\n' + 
                 '(OSINT-Enhanced - Text-Only)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    improvement_file = os.path.join(OUTPUT_DIR, "improvement_analysis.png")
    plt.savefig(improvement_file, dpi=300, bbox_inches='tight')
    print(f"💾 Improvement analysis saved: {improvement_file}")
    plt.close()

# ========================================
# FEATURE COUNT COMPARISON
# ========================================

def plot_feature_comparison(results):
    """Plot feature count comparison."""
    
    modes = []
    feature_counts = []
    accuracies = []
    
    for mode in ['TEXT_ONLY', 'OSINT_ENHANCED']:
        if mode in results:
            modes.append('Text-Only' if mode == 'TEXT_ONLY' else 'OSINT-Enhanced')
            feature_counts.append(results[mode]['n_features'])
            accuracies.append(results[mode]['accuracy'] * 100)
    
    if not modes:
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(modes))
    width = 0.35
    
    # Bar plot for feature count
    ax1.bar(x - width/2, feature_counts, width, label='Feature Count', 
            color='#3498db', alpha=0.7)
    ax1.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Features', fontsize=12, fontweight='bold', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    
    # Line plot for accuracy
    ax2 = ax1.twinx()
    ax2.plot(x, accuracies, color='#2ecc71', marker='o', linewidth=2, 
             markersize=10, label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.set_ylim([90, 100])
    
    # Add value labels
    for i, (fc, acc) in enumerate(zip(feature_counts, accuracies)):
        ax1.text(i - width/2, fc + 0.5, str(fc), ha='center', va='bottom', fontsize=10)
        ax2.text(i, acc + 0.3, f'{acc:.2f}%', ha='center', va='bottom', 
                fontsize=10, color='#2ecc71', fontweight='bold')
    
    plt.title('Feature Count vs Accuracy\n' + 
              'Impact of OSINT Features on Model Performance', 
              fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    feature_file = os.path.join(OUTPUT_DIR, "feature_comparison.png")
    plt.savefig(feature_file, dpi=300, bbox_inches='tight')
    print(f"💾 Feature comparison saved: {feature_file}")
    plt.close()

# ========================================
# GENERATE SUMMARY REPORT
# ========================================

def generate_summary_report(results, df_comparison):
    """Generate markdown summary report."""
    
    report_file = os.path.join(OUTPUT_DIR, "COMPARISON_REPORT.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Model Comparison Report: XGBoost Text-Only vs OSINT-Enhanced\n\n")
        f.write("## 📄 Reference Paper\n\n")
        f.write("**An et al. (2025)**  \n")
        f.write("*Multilingual Email Phishing Attacks Detection using OSINT and Machine Learning*  \n")
        f.write("🔗 https://arxiv.org/html/2501.08723v1\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 Performance Comparison\n\n")
        f.write("### Results Table\n\n")
        f.write(df_comparison.to_markdown(index=False))
        f.write("\n\n")
        
        # Improvement section
        if 'TEXT_ONLY' in results and 'OSINT_ENHANCED' in results:
            f.write("### 📈 OSINT Enhancement Impact\n\n")
            
            metrics_comparison = {
                'Accuracy': (results['OSINT_ENHANCED']['accuracy'] - results['TEXT_ONLY']['accuracy']) * 100,
                'Precision': (results['OSINT_ENHANCED']['precision'] - results['TEXT_ONLY']['precision']) * 100,
                'Recall': (results['OSINT_ENHANCED']['recall'] - results['TEXT_ONLY']['recall']) * 100,
                'F1 Score': (results['OSINT_ENHANCED']['f1_score'] - results['TEXT_ONLY']['f1_score']) * 100
            }
            
            for metric, improvement in metrics_comparison.items():
                f.write(f"- **{metric}**: {improvement:+.2f}%\n")
            
            f.write("\n")
            
            # Paper comparison
            f.write("### 📚 Comparison with Paper\n\n")
            f.write("| Metric | Our Result | Paper Result |\n")
            f.write("|--------|------------|-------------|\n")
            f.write(f"| XGBoost Text-Only Accuracy | {results['TEXT_ONLY']['accuracy']*100:.2f}% | 95.39% |\n")
            f.write(f"| XGBoost + OSINT Accuracy | {results['OSINT_ENHANCED']['accuracy']*100:.2f}% | 96.71% |\n")
            f.write(f"| Improvement | {metrics_comparison['Accuracy']:+.2f}% | +1.32% |\n")
            f.write("\n")
        
        # Confusion matrices
        f.write("---\n\n")
        f.write("## 🔍 Confusion Matrix Analysis\n\n")
        
        for mode in ['TEXT_ONLY', 'OSINT_ENHANCED']:
            if mode in results:
                cm = np.array(results[mode]['confusion_matrix'])
                tn, fp, fn, tp = cm.ravel()
                
                model_name = 'Text-Only' if mode == 'TEXT_ONLY' else 'OSINT-Enhanced'
                f.write(f"### {model_name}\n\n")
                f.write(f"```\n")
                f.write(f"              Predicted\n")
                f.write(f"              Safe  Phishing\n")
                f.write(f"Actual Safe     {tn:4d}    {fp:4d}\n")
                f.write(f"       Phishing {fn:4d}    {tp:4d}\n")
                f.write(f"```\n\n")
                f.write(f"- True Negatives (TN): {tn}\n")
                f.write(f"- False Positives (FP): {fp}\n")
                f.write(f"- False Negatives (FN): {fn}\n")
                f.write(f"- True Positives (TP): {tp}\n\n")
        
        # Features
        f.write("---\n\n")
        f.write("## 🎯 Feature Analysis\n\n")
        
        if 'TEXT_ONLY' in results:
            f.write(f"### Text-Only Features ({results['TEXT_ONLY']['n_features']} features)\n\n")
            for feat in results['TEXT_ONLY']['feature_names']:
                f.write(f"- `{feat}`\n")
            f.write("\n")
        
        if 'OSINT_ENHANCED' in results:
            text_features = results['TEXT_ONLY']['feature_names'] if 'TEXT_ONLY' in results else []
            osint_features = [f for f in results['OSINT_ENHANCED']['feature_names'] if f not in text_features]
            
            f.write(f"### OSINT-Enhanced Features ({results['OSINT_ENHANCED']['n_features']} features)\n\n")
            f.write(f"**Text Features** ({len(text_features)} features):\n")
            for feat in text_features:
                f.write(f"- `{feat}`\n")
            f.write(f"\n**OSINT Features** ({len(osint_features)} features):\n")
            for feat in osint_features:
                f.write(f"- `{feat}`\n")
            f.write("\n")
        
        # Conclusion
        f.write("---\n\n")
        f.write("## 💡 Key Findings\n\n")
        
        if 'TEXT_ONLY' in results and 'OSINT_ENHANCED' in results:
            acc_improvement = (results['OSINT_ENHANCED']['accuracy'] - results['TEXT_ONLY']['accuracy']) * 100
            
            if acc_improvement > 0:
                f.write(f"1. **OSINT features improve model accuracy by {acc_improvement:.2f}%**\n")
                f.write(f"2. Paper reported +1.32% improvement, our result shows {acc_improvement:.2f}%\n")
                f.write(f"3. Both results confirm OSINT enhancement benefits\n")
            else:
                f.write(f"1. OSINT features did not improve accuracy in this experiment\n")
                f.write(f"2. Possible reasons: dataset size, feature extraction quality, or random variation\n")
        
        f.write("\n---\n\n")
        f.write("## 📁 Generated Files\n\n")
        f.write("- `comparison_table.csv` - Metrics comparison table\n")
        f.write("- `confusion_matrices.png` - Confusion matrix visualization\n")
        f.write("- `metrics_comparison.png` - Performance metrics bar chart\n")
        f.write("- `improvement_analysis.png` - Improvement analysis\n")
        f.write("- `feature_comparison.png` - Feature count vs accuracy\n")
        f.write("- `COMPARISON_REPORT.md` - This report\n\n")
        
        f.write("---\n\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f"💾 Summary report saved: {report_file}")

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    print("="*80)
    print("  📊 MODEL COMPARISON - XGBoost Text-Only vs OSINT-Enhanced")
    print("="*80)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    print(f"\n✅ Loaded results for: {', '.join(results.keys())}")
    
    # Generate comparison table
    df_comparison = generate_comparison_table(results)
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    plot_confusion_matrices(results)
    plot_metrics_comparison(results)
    plot_improvement_analysis(results)
    plot_feature_comparison(results)
    
    # Generate summary report
    print(f"\n📄 Generating summary report...")
    generate_summary_report(results, df_comparison)
    
    print(f"\n{'='*80}")
    print(f"  ✅ COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 All files saved to: {OUTPUT_DIR}/")
    print(f"\n💡 Review:")
    print(f"   - {OUTPUT_DIR}/COMPARISON_REPORT.md (summary)")
    print(f"   - {OUTPUT_DIR}/metrics_comparison.png (visualization)")
    print(f"   - {OUTPUT_DIR}/confusion_matrices.png (detailed analysis)")

if __name__ == "__main__":
    main()

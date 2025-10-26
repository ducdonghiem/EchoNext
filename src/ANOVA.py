import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set plotting style
sns.set_theme(style="whitegrid")

# Configuration
class Config:
    METADATA_PATH = "1.1.0/echonext_metadata_100k.csv"
    FEATURES_PATH = "extracted_ecg_features_train.csv"
    SPLIT_FILTER = 'train'
    K_BEST = 10
    OUTPUT_DIR = Path("feature_selection_results")
    
    # All target columns to analyze
    TARGET_COLUMNS = [
        'lvef_lte_45_flag',
        'lvwt_gte_13_flag',
        'aortic_stenosis_moderate_or_greater_flag',
        'aortic_regurgitation_moderate_or_greater_flag',
        'mitral_regurgitation_moderate_or_greater_flag',
        'tricuspid_regurgitation_moderate_or_greater_flag',
        'pulmonary_regurgitation_moderate_or_greater_flag',
        'rv_systolic_dysfunction_moderate_or_greater_flag',
        'pericardial_effusion_moderate_large_flag',
        'pasp_gte_45_flag',
        'tr_max_gte_32_flag',
        'shd_moderate_or_greater_flag'
    ]
    
    COLS_TO_DROP = [
        "HRV_DFA_alpha1", "Morph_R_Amp_Mean", "Morph_R_Amp_Std",
        "Morph_T_R_Ratio", "HRV_LF_HF_Ratio", "HFn", "LFn"
    ]
    
    ROWS_TO_REMOVE = [69,73,435,596,841,1028,1715,2144,2226,3393,3703,4161,4261,
                      7135,7739,11696,11933,12349,13528,13806,15558,15609,16110,
                      16379,17357,17692,17967,18008,18724,19590,19656,19690,20112,
                      20482,21009,21453,21689,21977,24520,25711,25776,26053,28270,
                      28808,28897,28924,30113,30505,30630,30902,30971,31268,32013,
                      32275,32632,33540,33906,34853,34905,35127,35353,36459,37090,
                      37801,39115,40293,40632,41084,41252,43533,43576,45367,45747,
                      46283,46860,48047,48463,49441,50626,52345,52692,53166,53686,
                      54204,56437,56471,58298,58860,61065,61408,63118,64127,65950,
                      67954,69575,70462,71894]


def setup_output_directories(base_dir: Path):
    """Create organized directory structure for outputs."""
    dirs = {
        'base': base_dir,
        'rankings': base_dir / 'rankings',
        'plots': base_dir / 'plots',
        'distributions': base_dir / 'distributions',
        'summary': base_dir / 'summary'
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_and_preprocess_data(config: Config):
    """Load and preprocess features and metadata."""
    print("="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load metadata
    print(f"Loading metadata from: {config.METADATA_PATH}")
    metadata = pd.read_csv(config.METADATA_PATH)
    
    # Filter to train split
    print(f"Filtering metadata for split='{config.SPLIT_FILTER}'...")
    train_metadata = metadata[metadata['split'] == config.SPLIT_FILTER].reset_index(drop=True)
    
    # Load features
    print(f"Loading features from: {config.FEATURES_PATH}")
    X = pd.read_csv(config.FEATURES_PATH)
    
    # Drop problematic columns
    cols_to_drop_present = [c for c in config.COLS_TO_DROP if c in X.columns]
    if cols_to_drop_present:
        print(f"Dropping {len(cols_to_drop_present)} high-missing columns")
        X = X.drop(columns=cols_to_drop_present)
    
    # Check for and handle infinite values BEFORE imputation
    print("Checking for infinite/extreme values...")
    inf_mask = np.isinf(X.values)
    if inf_mask.any():
        n_inf = inf_mask.sum()
        print(f"⚠️  Found {n_inf} infinite values. Replacing with NaN...")
        X = X.replace([np.inf, -np.inf], np.nan)
    
    # Check for extremely large values (potential overflow)
    max_val = np.finfo(np.float64).max / 1e10  # Conservative threshold
    extreme_mask = np.abs(X.values) > max_val
    if np.any(extreme_mask):
        n_extreme = np.sum(extreme_mask)
        print(f"⚠️  Found {n_extreme} extremely large values. Replacing with NaN...")
        X = X.mask(np.abs(X) > max_val, np.nan)
    
    # Report missing data before imputation
    missing_pct = (X.isna().sum() / len(X) * 100).sort_values(ascending=False)
    cols_with_missing = missing_pct[missing_pct > 0]
    if len(cols_with_missing) > 0:
        print(f"\nColumns with missing data: {len(cols_with_missing)}")
        print(f"Total missing values: {X.isna().sum().sum()}")
        if len(cols_with_missing) <= 10:
            for col, pct in cols_with_missing.items():
                print(f"  - {col}: {pct:.2f}%")
    
    # Impute missing values
    print("\nImputing missing values with median strategy...")
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Remove problematic rows
    print(f"Removing {len(config.ROWS_TO_REMOVE)} problematic rows...")
    py_idx = sorted({r-1 for r in config.ROWS_TO_REMOVE})
    mask = ~X.index.isin(py_idx)
    X = X.loc[mask].reset_index(drop=True)
    train_metadata = train_metadata.loc[mask].reset_index(drop=True)
    
    # Final check for any remaining issues
    if X.isna().any().any():
        print("⚠️  WARNING: NaN values still present after imputation!")
        X = X.fillna(0)  # Fallback
    
    if np.isinf(X.values).any():
        print("⚠️  WARNING: Infinite values still present!")
        X = X.replace([np.inf, -np.inf], 0)  # Fallback
    
    # Scale features
    print("Scaling features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"\nFinal dataset: {len(X)} samples × {X.shape[1]} features")
    return X, X_scaled, train_metadata, scaler


def rank_features_with_anova(X_scaled: pd.DataFrame, y: pd.Series, k: int = 10):
    """Rank features using ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=min(k, X_scaled.shape[1]))
    selector.fit(X_scaled, y)
    
    feature_ranking = pd.DataFrame({
        'Feature': X_scaled.columns,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    }).sort_values(by='F_Score', ascending=False).reset_index(drop=True)
    
    top_k_features = feature_ranking.head(k)['Feature'].tolist()
    return feature_ranking, top_k_features


def plot_feature_ranking(ranking_df: pd.DataFrame, target_name: str, 
                         output_path: Path, k_best: int):
    """Visualize F-Score rankings."""
    df_plot = ranking_df.head(30)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='F_Score', y='Feature', data=df_plot, hue='Feature', 
                palette="viridis", legend=False, ax=ax)
    ax.set_title(f'Feature Importance for {target_name}\n(ANOVA F-Score)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('F-Score (Higher is better)', fontsize=12)
    ax.set_ylabel('Feature Name', fontsize=12)
    
    if len(df_plot) > k_best:
        ax.axhline(y=k_best - 0.5, color='red', linestyle='--', 
                   linewidth=2, label=f'Top {k_best} Cutoff')
        ax.legend()
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_distributions(X_original: pd.DataFrame, y: pd.Series, 
                               top_features: list, ranking_df: pd.DataFrame,
                               target_name: str, output_path: Path):
    """Generate box plots for top features showing class separation."""
    plot_data = X_original[top_features].copy()
    plot_data['Label'] = y.values
    plot_data['Label'] = plot_data['Label'].map({0: 'Negative', 1: 'Positive'})
    
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    for i, feature in enumerate(top_features):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x='Label', y=feature, data=plot_data, hue='Label',
                   palette={'Negative': 'skyblue', 'Positive': 'salmon'}, 
                   legend=False, ax=ax)
        
        f_score = ranking_df[ranking_df['Feature'] == feature]['F_Score'].iloc[0]
        ax.set_title(f'{feature}\n(F-Score: {f_score:.2f})', fontsize=11, fontweight='bold')
        ax.set_xlabel(target_name, fontsize=10)
        ax.set_ylabel('Feature Value', fontsize=10)
    
    plt.suptitle(f'Top {len(top_features)} Features for {target_name}', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_explanation_table(X: pd.DataFrame, y: pd.Series, 
                            top_features: list, ranking_df: pd.DataFrame):
    """Create detailed statistics table for top features."""
    top_X = X[top_features].copy()
    top_X['Label'] = y.values
    
    # Calculate class means
    mean_stats = top_X.groupby('Label').mean().transpose()
    mean_stats.columns = ['Mean_Negative', 'Mean_Positive']
    mean_stats['Abs_Difference'] = np.abs(
        mean_stats['Mean_Positive'] - mean_stats['Mean_Negative']
    )
    
    # Add class standard deviations
    std_stats = top_X.groupby('Label').std().transpose()
    std_stats.columns = ['Std_Negative', 'Std_Positive']
    
    # Merge with F-scores
    stats_df = ranking_df[ranking_df['Feature'].isin(top_features)].set_index('Feature')
    explanation_df = stats_df.merge(mean_stats, left_index=True, right_index=True)
    explanation_df = explanation_df.merge(std_stats, left_index=True, right_index=True)
    
    # Calculate effect size (Cohen's d approximation)
    pooled_std = np.sqrt((explanation_df['Std_Negative']**2 + 
                          explanation_df['Std_Positive']**2) / 2)
    explanation_df['Effect_Size'] = explanation_df['Abs_Difference'] / pooled_std
    
    return explanation_df.sort_values(by='F_Score', ascending=False)


def analyze_target(X: pd.DataFrame, X_scaled: pd.DataFrame, y: pd.Series,
                  target_name: str, config: Config, dirs: dict):
    """Perform complete analysis for a single target."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {target_name}")
    print(f"{'='*80}")
    
    # Check class balance
    class_counts = y.value_counts()
    positive_pct = (class_counts.get(1, 0) / len(y)) * 100
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Positive rate: {positive_pct:.2f}%")
    
    # Skip if insufficient positive samples
    if class_counts.get(1, 0) < 10:
        print(f"⚠️  WARNING: Too few positive samples ({class_counts.get(1, 0)}). Skipping.")
        return None
    
    # Rank features
    print(f"Ranking features (k={config.K_BEST})...")
    ranking_df, top_features = rank_features_with_anova(X_scaled, y, k=config.K_BEST)
    
    # Save ranking CSV
    ranking_path = dirs['rankings'] / f"{target_name}_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    print(f"✓ Saved ranking to: {ranking_path}")
    
    # Plot F-score ranking
    plot_path = dirs['plots'] / f"{target_name}_fscores.png"
    plot_feature_ranking(ranking_df, target_name, plot_path, config.K_BEST)
    print(f"✓ Saved F-score plot to: {plot_path}")
    
    # Plot distributions
    dist_path = dirs['distributions'] / f"{target_name}_distributions.png"
    plot_feature_distributions(X, y, top_features, ranking_df, target_name, dist_path)
    print(f"✓ Saved distribution plot to: {dist_path}")
    
    # Create explanation table
    explanation_df = create_explanation_table(X, y, top_features, ranking_df)
    
    # Save explanation table
    table_path = dirs['rankings'] / f"{target_name}_explanation.csv"
    explanation_df.to_csv(table_path)
    print(f"✓ Saved explanation table to: {table_path}")
    
    return {
        'target': target_name,
        'n_samples': len(y),
        'positive_rate': positive_pct,
        'top_features': top_features,
        'top_f_scores': ranking_df.head(config.K_BEST)['F_Score'].tolist(),
        'mean_top_f_score': ranking_df.head(config.K_BEST)['F_Score'].mean()
    }


def create_summary_report(all_results: list, dirs: dict, config: Config):
    """Create comprehensive summary report across all targets."""
    print(f"\n{'='*80}")
    print("CREATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    # Filter out None results
    valid_results = [r for r in all_results if r is not None]
    
    # 1. Summary statistics table
    summary_df = pd.DataFrame([{
        'Target': r['target'],
        'N_Samples': r['n_samples'],
        'Positive_Rate_%': r['positive_rate'],
        'Mean_Top_F_Score': r['mean_top_f_score'],
        'Max_F_Score': max(r['top_f_scores']),
        'Min_F_Score': min(r['top_f_scores'])
    } for r in valid_results])
    
    summary_path = dirs['summary'] / 'target_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved target summary to: {summary_path}")
    
    # 2. Feature frequency across targets
    feature_counter = {}
    for result in valid_results:
        for feature in result['top_features']:
            feature_counter[feature] = feature_counter.get(feature, 0) + 1
    
    feature_freq_df = pd.DataFrame([
        {'Feature': feat, 'Frequency': count, 'Percentage': (count/len(valid_results))*100}
        for feat, count in sorted(feature_counter.items(), key=lambda x: -x[1])
    ])
    
    freq_path = dirs['summary'] / 'feature_frequency.csv'
    feature_freq_df.to_csv(freq_path, index=False)
    print(f"✓ Saved feature frequency to: {freq_path}")
    
    # 3. Top features per target table
    top_features_dict = {r['target']: r['top_features'] for r in valid_results}
    
    with open(dirs['summary'] / 'top_features_per_target.json', 'w') as f:
        json.dump(top_features_dict, f, indent=2)
    print(f"✓ Saved top features JSON to: {dirs['summary'] / 'top_features_per_target.json'}")
    
    # 4. Create heatmap of feature importance across targets
    create_importance_heatmap(valid_results, dirs, config)
    
    # 5. Create markdown summary report
    create_markdown_report(valid_results, feature_freq_df, summary_df, dirs)
    
    return summary_df, feature_freq_df


def create_importance_heatmap(results: list, dirs: dict, config: Config):
    """Create heatmap showing feature importance across all targets."""
    # Build matrix: rows=features, cols=targets, values=F-scores
    all_features = set()
    for result in results:
        all_features.update(result['top_features'])
    
    # Load all ranking CSVs to get F-scores
    importance_matrix = []
    feature_list = sorted(all_features)
    
    for feature in feature_list:
        row = []
        for result in results:
            ranking_path = dirs['rankings'] / f"{result['target']}_ranking.csv"
            ranking_df = pd.read_csv(ranking_path)
            f_score = ranking_df[ranking_df['Feature'] == feature]['F_Score'].values
            row.append(f_score[0] if len(f_score) > 0 else 0)
        importance_matrix.append(row)
    
    importance_df = pd.DataFrame(
        importance_matrix,
        index=feature_list,
        columns=[r['target'] for r in results]
    )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(feature_list) * 0.3)))
    sns.heatmap(importance_df, annot=False, cmap='YlOrRd', cbar_kws={'label': 'F-Score'},
                linewidths=0.5, ax=ax)
    ax.set_title('Feature Importance Across All Targets (F-Scores)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Target Variable', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    heatmap_path = dirs['summary'] / 'feature_importance_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved importance heatmap to: {heatmap_path}")


def create_markdown_report(results: list, feature_freq_df: pd.DataFrame,
                          summary_df: pd.DataFrame, dirs: dict):
    """Create a markdown summary report."""
    report_path = dirs['summary'] / 'analysis_report.md'
    
    with open(report_path, 'w') as f:
        f.write(f"# Feature Selection Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Targets Analyzed:** {len(results)}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n")
        
        f.write("## Most Frequently Selected Features\n\n")
        f.write(feature_freq_df.head(20).to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n")
        
        f.write("## Top Features Per Target\n\n")
        for result in results:
            f.write(f"### {result['target']}\n\n")
            f.write(f"- **Positive Rate:** {result['positive_rate']:.2f}%\n")
            f.write(f"- **Mean F-Score (Top {len(result['top_features'])}):** "
                   f"{result['mean_top_f_score']:.2f}\n")
            f.write(f"- **Top Features:** {', '.join(result['top_features'][:5])}\n\n")
    
    print(f"✓ Saved markdown report to: {report_path}")


def main():
    """Main execution function."""
    config = Config()
    
    print("="*80)
    print("MULTI-TARGET ANOVA FEATURE SELECTION ANALYSIS")
    print("="*80)
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Number of targets: {len(config.TARGET_COLUMNS)}")
    print(f"K-best features: {config.K_BEST}")
    
    # Setup directories
    dirs = setup_output_directories(config.OUTPUT_DIR)
    
    # Load and preprocess data
    X, X_scaled, metadata, scaler = load_and_preprocess_data(config)
    
    # Analyze each target
    all_results = []
    for target_col in config.TARGET_COLUMNS:
        if target_col not in metadata.columns:
            print(f"\n⚠️  WARNING: '{target_col}' not found in metadata. Skipping.")
            continue
        
        y = metadata[target_col]
        result = analyze_target(X, X_scaled, y, target_col, config, dirs)
        all_results.append(result)
    
    # Create summary report
    summary_df, feature_freq_df = create_summary_report(all_results, dirs, config)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ All results saved to: {config.OUTPUT_DIR}")
    print(f"✓ Total targets analyzed: {len([r for r in all_results if r is not None])}")
    print(f"\nTop 5 most frequently selected features:")
    print(feature_freq_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
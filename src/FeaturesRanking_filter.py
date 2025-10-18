import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
# New Imports for Explainability and Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style for better readability
sns.set_theme(style="whitegrid")

def rank_features_with_anova(X: pd.DataFrame, y: pd.Series, k: int = 10):
    """
    Ranks continuous features for a binary classification task using the 
    ANOVA F-test (f_classif) and selects the top 'k' features.
    
    Note: X should be scaled before calling this function for robust ranking, 
    but the unscaled X must be used for meaningful plotting/explanation.
    """
    print(f"--- Starting ANOVA F-test Feature Selection (k={k}) ---")

    # 1. Initialize the selector
    selector = SelectKBest(score_func=f_classif, k=k)

    # 2. Fit the selector to the data
    selector.fit(X, y)

    # 3. Extract scores and p-values
    f_scores = selector.scores_
    p_values = selector.pvalues_

    # 4. Create a DataFrame for easy ranking
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': f_scores,
        'P_Value': p_values
    })

    # 5. Sort the features by F_Score (highest F-score is best)
    feature_ranking = feature_ranking.sort_values(by='F_Score', ascending=False).reset_index(drop=True)

    # 6. Select the top k features
    top_k_features = feature_ranking.head(k)['Feature'].tolist()
    
    print("\n[Ranking Complete]")
    print(f"Top {k} features selected: {top_k_features}")
    
    return feature_ranking, top_k_features

def plot_feature_ranking(ranking_df: pd.DataFrame):
    """Visualizes the sorted F-Scores to show feature importance drop-off."""
    print("\n--- Generating F-Score Ranking Plot ---")
    
    # Visualize top 30 features to see the tail of importance
    df_plot = ranking_df.head(30)
    
    plt.figure(figsize=(10, 8))
    # Use a color palette to distinguish the top K features if desired
    bar_plot = sns.barplot(x='F_Score', y='Feature', data=df_plot, palette="viridis")
    plt.title('Feature Importance (ANOVA F-Score)', fontsize=14)
    plt.xlabel('F-Score (Higher is better)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    # Highlight the separation between the top features and the rest
    if len(df_plot) > K_BEST:
         plt.axhline(y=K_BEST - 0.5, color='red', linestyle='--', linewidth=1, label=f'Top {K_BEST} Cutoff')
         plt.legend()
    plt.gca().invert_yaxis() # Highest score on top
    plt.tight_layout()
    plt.show()
    print("F-Score ranking plot generated. (Shows overall feature contribution)")

def plot_feature_distributions(X_original: pd.DataFrame, y: pd.Series, top_features: list, ranking_df: pd.DataFrame):
    """
    Generates box plots for top features to visually explain class separation.
    
    The wide separation between the box plot centers is the visual explanation for 
    the high F-Score.
    """
    print("\n--- Generating Feature Distribution Plots (Visual Explanation) ---")
    
    # Combine original features and label for plotting
    plot_data = X_original[top_features].copy()
    plot_data['SHD_Label'] = y.values
    plot_data['SHD_Label'] = plot_data['SHD_Label'].map({0: 'SHD Negative', 1: 'SHD Positive'})
    
    n_features = len(top_features)
    # Determine grid size (e.g., 3 columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols 
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, feature in enumerate(top_features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x='SHD_Label', y=feature, data=plot_data, palette={'SHD Negative': 'skyblue', 'SHD Positive': 'salmon'})
        
        # Get the F-score for the title
        f_score = ranking_df[ranking_df['Feature'] == feature]['F_Score'].iloc[0]
        plt.title(f'{feature}\n(F-Score: {f_score:.2f})', fontsize=12)
        plt.xlabel('Structural Heart Disease Status', fontsize=10)
        plt.ylabel('Feature Value (Unscaled)', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    print("Box plots generated for top features. (Shows class separation)")

# -----------------------------------------------
# --- LOAD AND PREPARE DATA ---
# -----------------------------------------------
METADATA_PATH = "1.1.0/echonext_metadata_100k.csv"
FEATURES_PATH = "extracted_ecg_features_train.csv"
TARGET_COLUMN = 'shd_moderate_or_greater_flag'
SPLIT_FILTER = 'train'
K_BEST = 10 

try:
    print(f"Loading metadata from: {METADATA_PATH}")
    metadata = pd.read_csv(METADATA_PATH)

    # 1. Filter metadata to the 'train' split
    print(f"Filtering metadata for split='{SPLIT_FILTER}'...")
    train_metadata = metadata[metadata['split'] == SPLIT_FILTER].reset_index(drop=True)

    # 2. Extract the target variable (y)
    y = train_metadata[TARGET_COLUMN].rename('SHD_Label')

    # 3. Load the pre-extracted features (X)
    print(f"Loading features from: {FEATURES_PATH}")
    X = pd.read_csv(FEATURES_PATH)
    
    # 4. Alignment Check (Crucial)
    # Assuming that 'extracted_ecg_features_train.csv' is perfectly row-aligned 
    # with the 'train' subset of the metadata file.
    if len(X) != len(y):
        print("\nERROR: Sample count mismatch! Features and Labels do not align.")
        print(f"Features loaded: {len(X)} samples. Filtered Labels: {len(y)} samples.")
        print("Please ensure 'extracted_ecg_features_train.csv' contains only samples where split='train'.")
        raise ValueError("Data misalignment detected between X and y.")
        
    N_SAMPLES = len(X)
    N_FEATURES = X.shape[1]
    print(f"Successfully loaded N={N_SAMPLES} samples with M={N_FEATURES} features.")

    # --- Feature Scaling (MANDATORY for SelectKBest using F-test if features are of different scales) ---
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # --- EXECUTE RANKING ---
    # We rank using the scaled data (X_scaled)
    ranking_df, top_features = rank_features_with_anova(X_scaled, y, k=K_BEST)

    # -----------------------------------------------
    # --- EXPLAINABILITY AND VISUALIZATION ---
    # -----------------------------------------------

    # 1. Visualization 1: F-Score Plot
    plot_feature_ranking(ranking_df)

    # 2. Visualization 2: Distribution Plots
    # We use the original data (X) for interpretable plots
    plot_feature_distributions(X, y, top_features, ranking_df)

    # 3. Mean Comparison Table (Quantifying Separation)
    print("\n" + "="*70)
    print("EXPLAINABILITY TABLE: Top Features' Class Means and Separation")
    print("="*70)

    # Isolate the top features in the original data and combine with label
    top_X_original = X[top_features].copy()
    top_X_original['SHD_Label'] = y.values
    
    # Calculate mean of each feature grouped by the SHD label
    mean_diffs = top_X_original.groupby('SHD_Label').mean().transpose()
    mean_diffs.columns = ['Mean_SHD_0', 'Mean_SHD_1']
    
    # Calculate the absolute difference between the class means
    mean_diffs['Abs_Difference'] = np.abs(mean_diffs['Mean_SHD_1'] - mean_diffs['Mean_SHD_0'])
    
    # Merge statistics with the F-Scores for the final table
    stats_df = ranking_df[ranking_df['Feature'].isin(top_features)].set_index('Feature')
    final_explanation_df = stats_df.merge(mean_diffs, left_index=True, right_index=True)
    
    # Select and rename columns for clarity
    final_explanation_df = final_explanation_df[[
        'F_Score', 
        'P_Value', 
        'Mean_SHD_0', 
        'Mean_SHD_1', 
        'Abs_Difference'
    ]].sort_values(by='F_Score', ascending=False)
    
    # Print the final explanation table
    print(final_explanation_df.to_markdown(numalign="left", stralign="left", floatfmt=".4f"))

    print("\n\nSummary of Top Features for downstream modeling:")
    print(top_features)
    
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: One of the required files was not found.")
    print(f"Please check the paths: {e}")
except ValueError as e:
    print(f"\nFATAL ERROR: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred during data loading or processing: {e}")

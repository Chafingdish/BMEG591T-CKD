# ============================================================
#  STAGE 1 — Import Dataset
# ============================================================
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# Download the Chronic Kidney Disease dataset from https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
chronic_kidney_disease = fetch_ucirepo(id=336)

# Data 
X = chronic_kidney_disease.data.features   # 400 rows x 24 feature columns
y = chronic_kidney_disease.data.targets    # 400 rows x 1 label column ('classification')

# Metadata & variable info
print(chronic_kidney_disease.metadata)
print(chronic_kidney_disease.variables)

# Combine into one working DataFrame
df = pd.concat([X, y], axis=1)
df.columns = list(X.columns) + ['target']

print('\n--- Dataset loaded ---')
print('Shape:', df.shape)          # Expect: (400, 25)
print(df.head(3))                  # First 3 rows preview

# ============================================================
#  STAGE 2 — Explore the Data
# ============================================================

# Overall structure 
print('\n=== DATA TYPES AND NULLS ===')
print(df.dtypes)                      # data type of every column

# Count missing values per column
print('\n=== MISSING VALUES PER COLUMN ===')
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_summary = pd.DataFrame({'count': missing, 'percent': missing_pct})
print(missing_summary[missing_summary['count'] > 0])   # only show columns with gaps

# Class balance
print('\n=== CLASS DISTRIBUTION ===')
print(df['target'].value_counts())
# Expect roughly: ckd = 250, notckd = 150

# Basic statistics for numeric columns
print('\n=== DESCRIPTIVE STATS ===')
print(df.describe())

# ============================================================
#  STAGE 3 — Clean & Preprocess
# ============================================================
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Fix known formatting issues in this specific dataset
# Some cells contain tab characters or trailing spaces
df.replace({
    '\t': '', '\t?': ''
}, regex=True, inplace=True)
df['target'] = df['target'].str.strip()   # ← add this line to remove whitespace/tabs


# Standardise yes/no → 1/0
df.replace({'yes': 1, 'no': 0,
            'Yes': 1, 'No': 0,
            'ckd': 1, 'notckd': 0}, inplace=True)

# Define column groups 
# Numeric: continuous measurements
numeric_cols = ['age','bp','sg','al','su','bgr','bu','sc',
                'sod','pot','hemo','pcv','wbcc','rbcc']

# Categorical: text labels (already converted to 0/1 above where possible)
cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

# All feature columns together
feature_cols = numeric_cols + cat_cols

# Force numeric columns to float (they may still be text)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # non-numeric → NaN

# Encode remaining categorical columns
# Some cat columns still have text like 'normal'/'abnormal'
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)          # ensure string before encoding
    df[col] = pd.to_numeric(df[col], errors='coerce')  # try numeric first
    if df[col].isnull().any():              # if NaN remain, label-encode
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean target column
df['target'] = pd.to_numeric(df['target'], errors='coerce')
df.dropna(subset=['target'], inplace=True)   # drop rows without a label
df['target'] = df['target'].astype(int)

# Impute missing feature values with column mean ---
# Mean imputation is the standard baseline for clinical datasets
imputer = SimpleImputer(strategy='mean')
df[feature_cols] = imputer.fit_transform(df[feature_cols])

# Verify the result ---
print('Missing values after cleaning:')
print(df[feature_cols + ['target']].isnull().sum().sum(), '← should be 0')
print('Dataset size after cleaning:', df.shape)
print('Class counts:', df['target'].value_counts().to_dict())

# ============================================================
#  STAGE 4 — Logistic Regression Feature Filtering
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Prepare scaled X and y
X_all = df[feature_cols].values
y_all = df['target'].values.astype(int)

scaler_lr = StandardScaler()
X_scaled_all = scaler_lr.fit_transform(X_all)

#Fit Logistic Regression on ALL 24 features
lr = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
lr.fit(X_scaled_all, y_all)
# Extract coefficients (effect size)
coef_df = pd.DataFrame({
    'feature':     feature_cols,
    'coefficient': lr.coef_[0]            # positive = raises CKD risk
}).sort_values('coefficient', key=abs, ascending=False)

print('\n=== Logistic Regression Coefficients (effect size) ===')
print(coef_df.to_string(index=False))

# Use statsmodels for p-values (more rigorous) 
# Install if needed:  pip install statsmodels
try:
    import statsmodels.api as sm

    X_sm = sm.add_constant(X_scaled_all)   # adds intercept column
    logit_model = sm.Logit(y_all, X_sm)

    # Use 'bfgs' method + regularization (l1_wt=0 means pure L2 ridge penalty)
    # disp=0 suppresses the iteration log
    result = logit_model.fit_regularized(
        method='l1',
        alpha=0.1,        # regularization strength — reduces multicollinearity
        disp=0
    )

    # fit_regularized doesn't give p-values directly, so we bootstrap them
    # Instead, use coefficient magnitude as significance proxy
    coef_series = pd.Series(result.params[1:], index=feature_cols)  # skip intercept

    pval_df = pd.DataFrame({
        'feature':     feature_cols,
        'coef':        coef_series.values,
        'abs_coef':    coef_series.abs().values,
    }).sort_values('abs_coef', ascending=False)

    # Features with near-zero coefficients after regularization = not useful
    # Threshold: keep features whose |coefficient| > 0.01
    threshold = 0.01
    pval_df['significant'] = pval_df['abs_coef'] > threshold
    pval_df = pval_df.sort_values('abs_coef', ascending=False)

    print('\n=== Feature Importance after Regularized Logistic Regression ===')
    print(pval_df.to_string(index=False))

    significant_features = pval_df[pval_df['significant']]['feature'].tolist()
    print(f'\nKept {len(significant_features)} significant features:')
    print(significant_features)

except Exception as e:
    print(f'statsmodels error: {e}')
    print('Falling back to sklearn coefficient ranking...')

    # Fallback: use sklearn's built-in L2 regularization
    from sklearn.linear_model import LogisticRegression as LR
    lr_fallback = LR(max_iter=2000, C=0.1, random_state=42)  # C=0.1 = stronger regularization
    lr_fallback.fit(X_scaled_all, y_all)

    coef_df = pd.DataFrame({
        'feature':   feature_cols,
        'abs_coef':  abs(lr_fallback.coef_[0])
    }).sort_values('abs_coef', ascending=False)

    threshold = 0.01
    coef_df['significant'] = coef_df['abs_coef'] > threshold

    print(coef_df.to_string(index=False))
    significant_features = coef_df[coef_df['significant']]['feature'].tolist()
    print(f'\nKept {len(significant_features)} features.')
    
# Prepare filtered X for the stress tests 
X_filtered = df[significant_features].values
print(f'\nFiltered feature matrix shape: {X_filtered.shape}')

# ============================================================
#  STAGE 5 — Stress Test 1: RFE Feature Reduction
# ============================================================
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt

# Define models
# ANN: two hidden layers (64 neurons, then 32 neurons)
ann = MLPClassifier(hidden_layer_sizes=(64, 32),
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True)    # stops if no improvement

# KNN: 5 nearest neighbours
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Set up cross-validation and F1 scorer
# 10-fold stratified CV: preserves class balance in each fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='binary', pos_label=1)

# Rank features by importance using RFE
# Use Logistic Regression as the RFE base estimator (fast, stable ranker)
rfe_ranker = LogisticRegression(max_iter=2000, random_state=42)
rfe = RFE(estimator=rfe_ranker, n_features_to_select=1, step=1)
rfe.fit(X_filtered, y_all)

sig_feature_ranking = pd.Series(rfe.ranking_, index=significant_features)
sorted_sig_features = sig_feature_ranking.sort_values().index.tolist()

print('Feature ranking (1 = most important):')
print(sig_feature_ranking.sort_values().to_string())

# Define subset sizes to test 
n = len(significant_features)      # total significant features
subset_sizes = [n, max(n-3,5), max(n//2,4), min(5, n)]
subset_sizes = sorted(set(subset_sizes), reverse=True)  # unique, descending

subset_labels = [f'Top {s}' for s in subset_sizes]

# Evaluate both models on each subset
rfe_results = {'ANN': {}, 'KNN': {}}
rfe_std     = {'ANN': {}, 'KNN': {}}

for size, label in zip(subset_sizes, subset_labels):
    feats = sorted_sig_features[:size]
    X_sub = df[feats].values

    scaler = StandardScaler()
    X_sub_s = scaler.fit_transform(X_sub)

    ann_cv = cross_val_score(ann, X_sub_s, y_all, cv=cv, scoring=f1_scorer)
    knn_cv = cross_val_score(knn, X_sub_s, y_all, cv=cv, scoring=f1_scorer)

    rfe_results['ANN'][label] = ann_cv.mean()
    rfe_results['KNN'][label] = knn_cv.mean()
    rfe_std['ANN'][label]     = ann_cv.std()
    rfe_std['KNN'][label]     = knn_cv.std()

    print(f'{label:10s} | ANN F1: {ann_cv.mean():.4f} ±{ann_cv.std():.4f}| KNN F1: {knn_cv.mean():.4f} ±{knn_cv.std():.4f}')


# Plot 
ann_vals = [rfe_results['ANN'][l] for l in subset_labels]
knn_vals = [rfe_results['KNN'][l] for l in subset_labels]

plt.figure(figsize=(9, 5))
plt.plot(subset_labels, ann_vals, marker='o', label='ANN',
         color='steelblue', linewidth=2.5)
plt.plot(subset_labels, knn_vals, marker='s', label='KNN',
         color='coral',    linewidth=2.5)
plt.fill_between(subset_labels,
    [rfe_results['ANN'][l]-rfe_std['ANN'][l] for l in subset_labels],
    [rfe_results['ANN'][l]+rfe_std['ANN'][l] for l in subset_labels],
    alpha=0.15, color='steelblue')
plt.fill_between(subset_labels,
    [rfe_results['KNN'][l]-rfe_std['KNN'][l] for l in subset_labels],
    [rfe_results['KNN'][l]+rfe_std['KNN'][l] for l in subset_labels],
    alpha=0.15, color='coral')
plt.title('Stress Test 1 — RFE Feature Reduction\nF1-Score vs Feature Count',
          fontsize=13, fontweight='bold')
plt.xlabel('Feature Subset (most → fewest features)')
plt.ylabel('F1-Score (10-fold CV mean)')
plt.ylim(0.75, 1.02)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rfe_stress_test.png', dpi=150)
plt.show()
print('Saved: rfe_stress_test.png')

# ============================================================
#  STAGE 6 — Stress Test 2: Progressive Missing-Value Simulation
# ============================================================

# Use the filtered (significant) features for this test too
X_base = df[significant_features].values.copy()

missing_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
decay_results = {'ANN': [], 'KNN': []}
decay_std     = {'ANN': [], 'KNN': []}

np.random.seed(42)   # makes results reproducible

print('\n=== Stress Test 2: Data Decay ===')
print(f'  {"Missing %":>10}  {"ANN F1":>10}  {"KNN F1":>10}  {"Winner":>8}')
print('  ' + '-'*44)

for rate in missing_rates:
    X_decay = X_base.copy().astype(float)

    # Inject random NaNs at the specified rate
    if rate > 0:
        mask = np.random.rand(*X_decay.shape) < rate
        X_decay[mask] = np.nan

    # Re-impute (mean of each column, computed on this degraded version)
    imp = SimpleImputer(strategy='mean')
    X_imp = imp.fit_transform(X_decay)

    # Scale
    sc = StandardScaler()
    X_s = sc.fit_transform(X_imp)

    # Cross-validate
    ann_cv = cross_val_score(ann, X_s, y_all, cv=cv, scoring=f1_scorer)
    knn_cv = cross_val_score(knn, X_s, y_all, cv=cv, scoring=f1_scorer)

    decay_results['ANN'].append(ann_cv.mean())
    decay_results['KNN'].append(knn_cv.mean())
    decay_std['ANN'].append(ann_cv.std())
    decay_std['KNN'].append(knn_cv.std())

    winner = 'ANN' if ann_cv.mean() > knn_cv.mean() else 'KNN'
    print(f'  {int(rate*100):>9}%  {ann_cv.mean():>10.4f}  {knn_cv.mean():>10.4f}  {winner:>8}')

# --- Plot ---
pct_labels = [f'{int(r*100)}%' for r in missing_rates]

plt.figure(figsize=(9, 5))
plt.plot(pct_labels, decay_results['ANN'], marker='o', label='ANN',
         color='steelblue', linewidth=2.5)
plt.plot(pct_labels, decay_results['KNN'], marker='s', label='KNN',
         color='coral',    linewidth=2.5)
plt.fill_between(pct_labels,
    [decay_results['ANN'][i]-decay_std['ANN'][i] for i in range(len(missing_rates))],
    [decay_results['ANN'][i]+decay_std['ANN'][i] for i in range(len(missing_rates))],
    alpha=0.15, color='steelblue')
plt.fill_between(pct_labels,
    [decay_results['KNN'][i]-decay_std['KNN'][i] for i in range(len(missing_rates))],
    [decay_results['KNN'][i]+decay_std['KNN'][i] for i in range(len(missing_rates))],
    alpha=0.15, color='coral')
plt.title('Stress Test 2 — Data Decay Simulation\nF1-Score vs Missing Data Rate',
          fontsize=13, fontweight='bold')
plt.xlabel('Percentage of Values Randomly Removed')
plt.ylabel('F1-Score (10-fold CV mean)')
plt.ylim(0.75, 1.02)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('decay_stress_test.png', dpi=150)
plt.show()
print('Saved: decay_stress_test.png')

# ============================================================
#  STAGE 7 — Final Summary & Comparison Chart
# ============================================================
import seaborn as sns

# Summary table: RFE stress test
print('\n' + '='*62)
print('RESULT TABLE 1 — RFE Stress Test (10-fold CV F1-Score)')
print('='*62)
print(f"  {'Subset':<12} {'ANN Mean':>10} {'ANN Std':>9} {'KNN Mean':>10} {'KNN Std':>9} {'Winner':>8}")
print('  ' + '-'*58)
for label in subset_labels:
    a  = rfe_results['ANN'][label]
    as_ = rfe_std['ANN'][label]
    k  = rfe_results['KNN'][label]
    ks = rfe_std['KNN'][label]
    w  = 'ANN' if a > k else 'KNN'
    print(f'  {label:<12} {a:>10.4f} {as_:>9.4f} {k:>10.4f} {ks:>9.4f} {w:>8}')

# Summary table: Data decay stress test
print('\n' + '='*62)
print('RESULT TABLE 2 — Data Decay Stress Test (10-fold CV F1-Score)')
print('='*62)
print(f"  {'Missing':>8} {'ANN Mean':>10} {'ANN Std':>9} {'KNN Mean':>10} {'KNN Std':>9} {'Winner':>8}")
print('  ' + '-'*58)
for i, rate in enumerate(missing_rates):
    a  = decay_results['ANN'][i]
    as_ = decay_std['ANN'][i]
    k  = decay_results['KNN'][i]
    ks = decay_std['KNN'][i]
    w  = 'ANN' if a > k else 'KNN'
    print(f'  {int(rate*100):>7}%  {a:>10.4f} {as_:>9.4f} {k:>10.4f} {ks:>9.4f} {w:>8}')

# Combined side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left panel: RFE
ax1 = axes[0]
ax1.plot(subset_labels, [rfe_results['ANN'][l] for l in subset_labels],
         marker='o', label='ANN', color='steelblue', linewidth=2.5)
ax1.plot(subset_labels, [rfe_results['KNN'][l] for l in subset_labels],
         marker='s', label='KNN', color='coral',    linewidth=2.5)
ax1.set_title('Stress Test 1\nRFE Feature Reduction', fontweight='bold')
ax1.set_xlabel('Feature Subset'); ax1.set_ylabel('F1-Score')
ax1.set_ylim(0.75, 1.02); ax1.legend(); ax1.grid(True, alpha=0.3)

# Right panel: Decay
ax2 = axes[1]
ax2.plot(pct_labels, decay_results['ANN'],
         marker='o', label='ANN', color='steelblue', linewidth=2.5)
ax2.plot(pct_labels, decay_results['KNN'],
         marker='s', label='KNN', color='coral',    linewidth=2.5)
ax2.set_title('Stress Test 2\nData Decay Simulation', fontweight='bold')
ax2.set_xlabel('Missing Data Rate'); ax2.set_ylabel('F1-Score')
ax2.set_ylim(0.75, 1.02); ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle('ANN vs KNN — Clinical Robustness Under Data Stress',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nSaved: final_comparison.png')
print('\n✓ All done! Check your project folder for the three .png chart files.')

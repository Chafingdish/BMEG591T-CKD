# ============================================================
#  STAGE 1 — Import Dataset
# ============================================================
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import os

# Download the Chronic Kidney Disease dataset from https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
chronic_kidney_disease = fetch_ucirepo(id=336)

# Data 
X = chronic_kidney_disease.data.features  # 400 rows x 24 feature columns
y = chronic_kidney_disease.data.targets
print(chronic_kidney_disease.metadata)
print(chronic_kidney_disease.variables)

# Combine into one
df = pd.concat([X, y], axis=1)
df.columns = list(X.columns) + ['target']

figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures')
os.makedirs(figures_dir, exist_ok=True)
def fig_path(filename):
    return os.path.join(figures_dir, filename)

print('\n--- Dataset loaded ---')
print('Shape:', df.shape) # Expect: (400, 25)
print(df.head(3)) 

# ============================================================
#  STAGE 2 — Explore the Data
# ============================================================
print(df.dtypes)

# Count missing values per column
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_summary = pd.DataFrame({'count': missing, 'percent': missing_pct})
print(missing_summary[missing_summary['count'] > 0])   # only columns with gaps

# Class balance
print(df['target'].value_counts()) # Expect roughly: ckd = 250, notckd = 150

# Basic statistics for numeric columns
print(df.describe())

# ============================================================
#  STAGE 3 — Clean & Preprocess
# ============================================================
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Fix known formatting issues in this specific dataset
df.replace({
    '\t': '', '\t?': ''
}, regex=True, inplace=True)
df['target'] = df['target'].str.strip()

# Standardise yes/no → 1/0
df.replace({'yes': 1, 'no': 0,
            'Yes': 1, 'No': 0,
            'ckd': 1, 'notckd': 0}, inplace=True)

numeric_cols = ['age','bp','sg','al','su','bgr','bu','sc',
                'sod','pot','hemo','pcv','wbcc','rbcc']
cat_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

# All feature columns together
feature_cols = numeric_cols + cat_cols

# Force numeric columns to float
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') # non-numeric -> NaN

# Encode remaining categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)  # ensure string
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isnull().any(): # if NaN remain, label-encode
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['target'] = pd.to_numeric(df['target'], errors='coerce')
df.dropna(subset=['target'], inplace=True)
df['target'] = df['target'].astype(int)

# Impute missing feature values with column mean
imputer = SimpleImputer(strategy='mean')
df[feature_cols] = imputer.fit_transform(df[feature_cols])

# result
print('Missing values after cleaning:')
print(df[feature_cols + ['target']].isnull().sum().sum(), '← should be 0')
print('Dataset size after cleaning:', df.shape)
print('Class counts:', df['target'].value_counts().to_dict())

# ============================================================
#  STAGE 4 — Logistic Regression Feature Filtering
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare scaled X and y
X_all = df[feature_cols].values
y_all = df['target'].values.astype(int)

scaler_lr = StandardScaler()
X_scaled_all = scaler_lr.fit_transform(X_all)

# Fit Logistic Regression on ALL 24 features
lr = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
lr.fit(X_scaled_all, y_all)
# Extract coefficients
coef_df = pd.DataFrame({
    'feature':     feature_cols,
    'coefficient': lr.coef_[0] # positive = raises CKD risk
}).sort_values('coefficient', key=abs, ascending=False)

print('\n=== Logistic Regression Coefficients (effect size) ===')
print(coef_df.to_string(index=False))

# Use statsmodels for p-values
# pip install statsmodels
try:
    import statsmodels.api as sm

    X_sm = sm.add_constant(X_scaled_all) 
    logit_model = sm.Logit(y_all, X_sm)

    # Use 'bfgs' method + regularization (l1_wt=0 means pure L2 ridge penalty)
    # disp=0 suppresses the iteration log
    result = logit_model.fit_regularized(
        method='l1',
        alpha=0.1, # regularization strength, moderate-to-low penalty
        disp=0
    )

    # bootstrap
    coef_series = pd.Series(result.params[1:], index=feature_cols)  # skip intercept
    pval_df = pd.DataFrame({
        'feature':     feature_cols,
        'coef':        coef_series.values,
        'abs_coef':    coef_series.abs().values,
    }).sort_values('abs_coef', ascending=False)

    # Threshold:|coefficient| > 0.01
    threshold = 0.01
    pval_df['significant'] = pval_df['abs_coef'] > threshold
    pval_df = pval_df.sort_values('abs_coef', ascending=False)
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

# feature importance bar chart
plt.figure(figsize=(9, 5))
colors_bar = ['#2E75B6' if c > 0 else '#C0392B' for c in pval_df[pval_df['significant']]['coef']]
bars = plt.barh(
    pval_df[pval_df['significant']]['feature'],
    pval_df[pval_df['significant']]['coef'],
    color=colors_bar, edgecolor='white', height=0.6
)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Regularised Coefficient (L1 Logistic Regression)')
plt.title('Feature Importance in Regularised Logistic Regression\n'
          'Blue = increases CKD risk    Red = decreases CKD risk',
          fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path('feature_importance.png'), dpi=150)
plt.show()
print('Saved: feature_importance.png')

# ============================================================
#  STAGE 5 — Stress Test 1: RFE Feature Reduction
# ============================================================
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
import matplotlib.pyplot as plt

# ANN: two hidden layers (64 neurons, then 32 neurons)
ann = MLPClassifier(hidden_layer_sizes=(64, 32),
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True) # stops if no improvement

# KNN: 5 nearest neighbours
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Logistic Regression as baseline
lr_model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)

# cross-validation and F1 scorer
# 10-fold stratified CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='binary', pos_label=1)

# Rank features by importance using RFE
# Use Logistic Regression as the RFE base estimator
rfe_ranker = LogisticRegression(max_iter=2000, random_state=42)
rfe = RFE(estimator=rfe_ranker, n_features_to_select=1, step=1)
rfe.fit(X_filtered, y_all)

sig_feature_ranking = pd.Series(rfe.ranking_, index=significant_features)
sorted_sig_features = sig_feature_ranking.sort_values().index.tolist()

print('Feature ranking (1 = most important):')
print(sig_feature_ranking.sort_values().to_string())

# Define subset sizes to test 
n = len(significant_features) # total significant features
subset_sizes = [n, max(n-3,5), max(n//2,4), min(5, n)]
subset_sizes = sorted(set(subset_sizes), reverse=True)

subset_labels = [f'Top {s}' for s in subset_sizes]

# Evaluate both models on each subset
rfe_results = {'ANN': {}, 'KNN': {}, 'LR': {}}
rfe_std  = {'ANN': {}, 'KNN': {}, 'LR': {}}

for size, label in zip(subset_sizes, subset_labels):
    feats = sorted_sig_features[:size]
    X_sub = df[feats].values

    scaler = StandardScaler()
    X_sub_s = scaler.fit_transform(X_sub)

    ann_cv = cross_val_score(ann, X_sub_s, y_all, cv=cv, scoring=f1_scorer)
    knn_cv = cross_val_score(knn, X_sub_s, y_all, cv=cv, scoring=f1_scorer)
    lr_cv = cross_val_score(lr_model, X_sub_s, y_all, cv=cv, scoring=f1_scorer)
        

    rfe_results['ANN'][label] = ann_cv.mean()
    rfe_results['KNN'][label] = knn_cv.mean()
    rfe_std['ANN'][label] = ann_cv.std()
    rfe_std['KNN'][label] = knn_cv.std()
    rfe_results['LR'][label] = lr_cv.mean()
    rfe_std['LR'][label] = lr_cv.std()
    
    print(f'{label:10s} | ANN F1: {ann_cv.mean():.4f} ±{ann_cv.std():.4f}| KNN F1: {knn_cv.mean():.4f} ±{knn_cv.std():.4f}| LR F1: {lr_cv.mean():.4f} ±{lr_cv.std():.4f}')


# Plot 
ann_vals = [rfe_results['ANN'][l] for l in subset_labels]
knn_vals = [rfe_results['KNN'][l] for l in subset_labels]
lr_vals = [rfe_results['LR'][l] for l in subset_labels]


plt.figure(figsize=(9, 5))
plt.plot(subset_labels, ann_vals, marker='o', label='ANN',  color='steelblue', linewidth=2.5)
plt.plot(subset_labels, knn_vals, marker='s', label='KNN',  color='coral',     linewidth=2.5)
plt.plot(subset_labels, lr_vals,  marker='^', label='LR',   color='seagreen',  linewidth=2.5)
plt.fill_between(subset_labels,
    [rfe_results['ANN'][l]-rfe_std['ANN'][l] for l in subset_labels],
    [rfe_results['ANN'][l]+rfe_std['ANN'][l] for l in subset_labels],
    alpha=0.12, color='steelblue')
plt.fill_between(subset_labels,
    [rfe_results['KNN'][l]-rfe_std['KNN'][l] for l in subset_labels],
    [rfe_results['KNN'][l]+rfe_std['KNN'][l] for l in subset_labels],
    alpha=0.12, color='coral')
plt.fill_between(subset_labels,
    [rfe_results['LR'][l]-rfe_std['LR'][l] for l in subset_labels],
    [rfe_results['LR'][l]+rfe_std['LR'][l] for l in subset_labels],
    alpha=0.12, color='seagreen')
plt.title('Stress Test 1 — RFE Feature Reduction\nF1-Score vs Feature Count',
          fontsize=13, fontweight='bold')
plt.xlabel('Feature Subset (most -> fewest features)')
plt.ylabel('F1-Score (10-fold CV mean)')
plt.ylim(0.75, 1.02)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path('rfe_stress_test.png'), dpi=150)
plt.show()

# ============================================================
#  STAGE 6 — Stress Test 2: Progressive Missing-Value Simulation
# ============================================================

# Use the filtered features for this test too
X_base = df[significant_features].values.copy()

missing_rates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
decay_results = {'ANN': [], 'KNN': [], 'LR': []}
decay_std     = {'ANN': [], 'KNN': [], 'LR': []}

np.random.seed(42)   # reproducible

print(f'  {"Missing %":>10}  {"ANN F1":>10}  {"KNN F1":>10}  {"Winner":>8}')
print('  ' + '-'*44)

for rate in missing_rates:
    X_decay = X_base.copy().astype(float)

    # Inject random NaNs
    if rate > 0:
        mask = np.random.rand(*X_decay.shape) < rate
        X_decay[mask] = np.nan

    # Re-impute
    imp = SimpleImputer(strategy='mean')
    X_imp = imp.fit_transform(X_decay)

    # Scale
    sc = StandardScaler()
    X_s = sc.fit_transform(X_imp)

    # Cross-validate
    ann_cv = cross_val_score(ann, X_s, y_all, cv=cv, scoring=f1_scorer)
    knn_cv = cross_val_score(knn, X_s, y_all, cv=cv, scoring=f1_scorer)
    lr_cv = cross_val_score(lr_model, X_s, y_all, cv=cv, scoring=f1_scorer)

    decay_results['ANN'].append(ann_cv.mean())
    decay_results['KNN'].append(knn_cv.mean())
    decay_std['ANN'].append(ann_cv.std())
    decay_std['KNN'].append(knn_cv.std())
    decay_results['LR'].append(lr_cv.mean())
    decay_std['LR'].append(lr_cv.std())
    
    
    winner = max({'ANN': ann_cv.mean(), 'KNN': knn_cv.mean(), 'LR': lr_cv.mean()}, 
                 key=lambda k: {'ANN': ann_cv.mean(), 'KNN': knn_cv.mean(), 'LR': lr_cv.mean()}[k])
    print(f'  {int(rate*100):>9}%  {ann_cv.mean():>10.4f}  {knn_cv.mean():>10.4f}  {lr_cv.mean():>10.4f}  {winner:>8}')



# Plot
pct_labels = [f'{int(r*100)}%' for r in missing_rates]

plt.figure(figsize=(9, 5))
plt.plot(pct_labels, decay_results['ANN'], marker='o', label='ANN', color='steelblue', linewidth=2.5)
plt.plot(pct_labels, decay_results['KNN'], marker='s', label='KNN', color='coral',     linewidth=2.5)
plt.plot(pct_labels, decay_results['LR'],  marker='^', label='LR',  color='seagreen',  linewidth=2.5)
plt.fill_between(pct_labels,
    [decay_results['ANN'][i]-decay_std['ANN'][i] for i in range(len(missing_rates))],
    [decay_results['ANN'][i]+decay_std['ANN'][i] for i in range(len(missing_rates))],
    alpha=0.12, color='steelblue')
plt.fill_between(pct_labels,
    [decay_results['KNN'][i]-decay_std['KNN'][i] for i in range(len(missing_rates))],
    [decay_results['KNN'][i]+decay_std['KNN'][i] for i in range(len(missing_rates))],
    alpha=0.12, color='coral')
plt.fill_between(pct_labels,
    [decay_results['LR'][i]-decay_std['LR'][i] for i in range(len(missing_rates))],
    [decay_results['LR'][i]+decay_std['LR'][i] for i in range(len(missing_rates))],
    alpha=0.12, color='seagreen')
plt.title('Stress Test 2 — Data Decay Simulation\nF1-Score vs Missing Data Rate',
          fontsize=13, fontweight='bold')
plt.xlabel('Percentage of Values Randomly Removed')
plt.ylabel('F1-Score (10-fold CV mean)')
plt.ylim(0.75, 1.02)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path('decay_stress_test.png'), dpi=150)
plt.show()

# ============================================================
#  STAGE 7 — Final Summary & Comparison Chart
# ============================================================
import seaborn as sns

# Summary table: RFE stress test
print('\n' + '='*72)
print('RESULT TABLE 1 — RFE Stress Test (10-fold CV F1-Score)')
print('='*72)
print(f"  {'Subset':<12} {'ANN Mean':>10} {'ANN Std':>9} {'KNN Mean':>10} {'KNN Std':>9} {'LR Mean':>9} {'LR Std':>8} {'Winner':>8}")
print('  ' + '-'*68)
for label in subset_labels:
    a, as_ = rfe_results['ANN'][label], rfe_std['ANN'][label]
    k, ks  = rfe_results['KNN'][label], rfe_std['KNN'][label]
    l, ls  = rfe_results['LR'][label],  rfe_std['LR'][label]
    w = max({'ANN': a, 'KNN': k, 'LR': l}, key=lambda x: {'ANN': a, 'KNN': k, 'LR': l}[x])
    print(f'  {label:<12} {a:>10.4f} {as_:>9.4f} {k:>10.4f} {ks:>9.4f} {l:>9.4f} {ls:>8.4f} {w:>8}')

# Summary table: Data decay stress test
print('\n' + '='*72)
print('RESULT TABLE 2 — Data Decay Stress Test (10-fold CV F1-Score)')
print('='*72)
print(f"  {'Missing':>8} {'ANN Mean':>10} {'ANN Std':>9} {'KNN Mean':>10} {'KNN Std':>9} {'LR Mean':>9} {'LR Std':>8} {'Winner':>8}")
print('  ' + '-'*68)
for i, rate in enumerate(missing_rates):
    a, as_ = decay_results['ANN'][i], decay_std['ANN'][i]
    k, ks  = decay_results['KNN'][i], decay_std['KNN'][i]
    l, ls  = decay_results['LR'][i],  decay_std['LR'][i]
    w = max({'ANN': a, 'KNN': k, 'LR': l}, key=lambda x: {'ANN': a, 'KNN': k, 'LR': l}[x])
    print(f'  {int(rate*100):>7}%  {a:>10.4f} {as_:>9.4f} {k:>10.4f} {ks:>9.4f} {l:>9.4f} {ls:>8.4f} {w:>8}')

# Combined 3 figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax1, ax2 = axes

lr_vals = [rfe_results['LR'][l] for l in subset_labels]
ax1.plot(subset_labels, [rfe_results['ANN'][l] for l in subset_labels],
         marker='o', label='ANN', color='steelblue', linewidth=2.5)
ax1.plot(subset_labels, [rfe_results['KNN'][l] for l in subset_labels],
         marker='s', label='KNN', color='coral', linewidth=2.5)
ax1.plot(subset_labels, lr_vals,
         marker='^', label='LR',  color='seagreen', linewidth=2.5)
ax1.set_title('Stress Test 1\nRFE Feature Reduction', fontweight='bold')
ax1.set_xlabel('Feature Subset'); ax1.set_ylabel('F1-Score')
ax1.set_ylim(0.75, 1.02); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(pct_labels, decay_results['ANN'],
         marker='o', label='ANN', color='steelblue', linewidth=2.5)
ax2.plot(pct_labels, decay_results['KNN'],
         marker='s', label='KNN', color='coral', linewidth=2.5)
ax2.plot(pct_labels, decay_results['LR'],
         marker='^', label='LR',  color='seagreen', linewidth=2.5)
ax2.set_title('Stress Test 2\nData Decay Simulation', fontweight='bold')
ax2.set_xlabel('Missing Data Rate'); ax2.set_ylabel('F1-Score')
ax2.set_ylim(0.75, 1.02); ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle('ANN vs KNN vs LR — Clinical Robustness Under Data Stress',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(fig_path('final_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
#  STAGE 8 — ROC Curves + AUC + Confusion Matrices
# ============================================================
from sklearn.metrics import (roc_curve, auc, RocCurveDisplay,
                              confusion_matrix, ConfusionMatrixDisplay,
                              classification_report)
from sklearn.model_selection import StratifiedKFold

#  Use the FULL significant feature set (Top 16) at 0% missing
# a single 80/20 split
from sklearn.model_selection import train_test_split

X_vis = df[significant_features].values
scaler_vis = StandardScaler()
X_vis_s = scaler_vis.fit_transform(X_vis)

X_train, X_test, y_train, y_test = train_test_split(
    X_vis_s, y_all, test_size=0.2, random_state=42, stratify=y_all)

models = {
    'ANN': MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000,
                         random_state=42, early_stopping=True),
    'KNN': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'LR':  LogisticRegression(max_iter=2000, random_state=42, C=1.0),
}
colours = {'ANN': 'steelblue', 'KNN': 'coral', 'LR': 'seagreen'}

# ROC Curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name}  (AUC = {roc_auc:.3f})',
            color=colours[name], linewidth=2.5)

ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random classifier')
ax.set_xlabel('False Positive Rate (1 – Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('ROC Curves — ANN vs KNN vs LR\n(80/20 stratified split)',
             fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_path('roc_curves.png'), dpi=150)
plt.show()
print('Saved: roc_curves.png')

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not CKD', 'CKD'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}', fontweight='bold', fontsize=13)

    # Annotate false negatives explicitly
    fn = cm[1, 0]
    ax.text(0, 1, f'FN={fn}\n(missed CKD)', ha='center', va='center',
            fontsize=9, color='red', fontweight='bold')

fig.suptitle('Confusion Matrices — Baseline (0% missing)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_path('confusion_matrices.png'), dpi=150)
plt.show()
print('Saved: confusion_matrices.png')

# Print
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'\n=== {name} Classification Report ===')
    print(classification_report(y_test, y_pred, target_names=['Not CKD', 'CKD']))

print('  rfe_stress_test.png, decay_stress_test.png, final_comparison.png,')
print('  roc_curves.png, confusion_matrices.png')
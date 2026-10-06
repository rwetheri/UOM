# -*- coding: utf-8 -*-
"""prediction.py

Can nasalance/duration acoustics alone tell you which phonological category
(nasal-target vs. oral-target) a word belongs to, without being told the target?
"""

import json
import os
import ssl
import sys
import urllib.request
from io import StringIO
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve, classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from collections import Counter
from matplotlib.ticker import PercentFormatter

try:
    from IPython.display import display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False

# Prenasalization measures that are structurally absent for many plain 'V' words.
PN_FEATURE_COLS = ['PN_CAvg', 'PN_CSD', 'PNCAvg3', 'PNCSD3']

CLEAN_FEATURE_COLS = [
    'WdDur', 'VDur', 'WdAvg', 'PN_CAvg', 'VAvg', 'WdSD',
    'PN_CSD', 'VSD', 'PNCAvg3', 'PNCSD3', 'VAvg3', 'VSD3',
]

# Acoustic-only set that avoids structurally missing prenazal measures.
NO_PN_FEATURE_COLS = [c for c in CLEAN_FEATURE_COLS if c not in PN_FEATURE_COLS]

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', context='talk')
PALETTE = sns.color_palette('viridis', 6)

SPEAKER_COL_CANDIDATES = ['Speaker', 'e']
GROUP_COL_CANDIDATES = ['Group_1', 'Group_2', 'Group']

SVM_PARAM_GRID = [
    {'svc__kernel': ['linear'], 'svc__C': [0.01, 0.1, 1, 10, 100]},
    {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10, 100],
     'svc__gamma': ['scale', 0.01, 0.1, 1]},
    {'svc__kernel': ['poly'], 'svc__degree': [2, 3], 'svc__C': [0.1, 1, 10]},
]

MULTICLASS_PARAM_GRID = [
    {'svc__kernel': ['linear'], 'svc__C': [0.01, 0.1, 1, 10, 100]},
    {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10, 100],
     'svc__gamma': ['scale', 0.01, 0.1, 1]},
]

# Metric columns reported as proportions (0-1) that should also get % columns.
PERCENT_METRIC_COLS = {
    'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'Macro F1',
    'Majority baseline', 'Chance baseline', 'accuracy', 'f1_macro',
    'majority_baseline', 'chance_baseline', 'precision', 'recall', 'f1-score',
}


def _pct(value, decimals=1):
    """Format a 0-1 proportion as a percentage string."""
    if pd.isna(value):
        return 'NA'
    return f'{float(value) * 100:.{decimals}f}%'


def _is_proportion_series(s):
    if not pd.api.types.is_numeric_dtype(s):
        return False
    vals = s.dropna()
    if vals.empty:
        return False
    return vals.between(0, 1).all()


def add_percentage_columns(df, metric_cols=None):
    """Add parallel '%' columns for metric proportions (keeps original decimals)."""
    out = df.copy()
    if metric_cols is None:
        metric_cols = [
            c for c in out.columns
            if c in PERCENT_METRIC_COLS or _is_proportion_series(out[c])
        ]
    for col in metric_cols:
        if col not in out.columns:
            continue
        pct_col = f'{col} %'
        out[pct_col] = out[col].map(lambda x: _pct(x))
    return out


def format_metrics_dict(metrics, decimals=1):
    """Format a metrics dict as 'name: value (pct%)' lines."""
    lines = []
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.floating)) and 0 <= value <= 1:
            lines.append(f'{name}: {value:.{decimals + 2}f} ({_pct(value, decimals)})')
        else:
            lines.append(f'{name}: {value}')
    return lines


def summarize_confusion_matrix(cm, labels=None):
    """Counts plus row/column/overall percentages for a confusion matrix."""
    cm = np.asarray(cm)
    labels = labels or [f'Class {i}' for i in range(cm.shape[0])]
    total = cm.sum()
    row_totals = cm.sum(axis=1, keepdims=True)
    col_totals = cm.sum(axis=0, keepdims=True)
    row_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    col_pct = np.divide(cm, col_totals, out=np.zeros_like(cm, dtype=float), where=col_totals != 0)

    detail_rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            detail_rows.append({
                'Actual': actual,
                'Predicted': predicted,
                'Count': int(cm[i, j]),
                '% of actual row': row_pct[i, j],
                '% of predicted col': col_pct[i, j],
                '% of all tokens': (cm[i, j] / total) if total else np.nan,
            })

    per_class = []
    for i, label in enumerate(labels):
        per_class.append({
            'Class': label,
            'N (actual)': int(row_totals[i, 0]),
            'Correct': int(cm[i, i]),
            'Recall %': row_pct[i, i] if row_totals[i, 0] else np.nan,
        })

    return {
        'counts': cm,
        'overall_accuracy': (np.trace(cm) / total) if total else np.nan,
        'detail': pd.DataFrame(detail_rows),
        'per_class_recall': pd.DataFrame(per_class),
    }


def print_metric_table(df, title=None, save_path=None):
    """Print a metric table with both decimal and % columns."""
    display_df = add_percentage_columns(df)
    if title:
        print(title)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(display_df.to_string(index=False))
    if save_path:
        display_df.to_csv(save_path, index=False)
        print(f'Saved: {save_path}')
    return display_df


def print_confusion_matrix_summary(cm, labels=None, title='Confusion matrix'):
    """Print counts and percentage breakdown for a confusion matrix."""
    summary = summarize_confusion_matrix(cm, labels=labels)
    print(f"\n{title}")
    print('Counts:')
    print(summary['counts'])
    print(
        f"Overall accuracy: {summary['overall_accuracy']:.3f} "
        f"({_pct(summary['overall_accuracy'])})"
    )
    print('\nPer-class recall (% of each actual class correctly predicted):')
    recall_df = summary['per_class_recall'].copy()
    recall_df['Recall %'] = recall_df['Recall %'].map(lambda x: _pct(x))
    print(recall_df.to_string(index=False))
    return summary


def _resolve_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a {label} column. Looked for: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def _n_group_splits(groups, max_splits=5):
    """Choose a GroupKFold split count that never exceeds the number of groups."""
    n_groups = pd.Series(groups).nunique()
    return max(2, min(max_splits, n_groups))


def _make_svm_pipe():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('svc', SVC(class_weight='balanced', probability=True, random_state=0)),
    ])


def _default_sheet_url():
    file_id = '1LPNTbrAX_ZMts7C1Q9s3vR0k4rmiXdXL'
    gid = '991931860'
    return f'https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}'


def read_csv_maybe_ssl(source):
    """Read a CSV URL, retrying with a permissive SSL context when needed."""
    if not (isinstance(source, str) and source.startswith('http')):
        return pd.read_csv(source)

    try:
        return pd.read_csv(source)
    except Exception as exc:
        msg = str(exc)
        if 'CERTIFICATE_VERIFY_FAILED' not in msg and 'ssl' not in msg.lower():
            raise

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(source, context=ctx) as response:
            text = response.read().decode('utf-8-sig')
        return pd.read_csv(StringIO(text))


def load_data(source, sheet='data_pared'):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str) and source.startswith('http'):
        df = read_csv_maybe_ssl(source)
    else:
        df = pd.read_excel(source, sheet_name=sheet)

    n_loaded = len(df)
    print(f"Loaded {n_loaded} rows. No rows are dropped for missing acoustic features.")

    # Keep rows marked Exclude? Reason. — they stay in the modeling sample.
    if 'Exclude? Reason.' in df.columns:
        n_exclude_marked = df['Exclude? Reason.'].notna().sum()
        if n_exclude_marked:
            print(
                f"Keeping {n_exclude_marked}/{n_loaded} rows "
                f"({_pct(n_exclude_marked / n_loaded)}) that have an Exclude? Reason. note."
            )

    # WordType is the class label. Tokens with no WordType cannot be trained/tested,
    # because they would be mis-coded as oral. Report them, then set them aside.
    n_no_wordtype = df['WordType'].isna().sum() if 'WordType' in df.columns else 0
    if n_no_wordtype:
        print(
            f"Setting aside {n_no_wordtype}/{n_loaded} rows "
            f"({_pct(n_no_wordtype / n_loaded)}) with no WordType label "
            f"(no nasal/oral class to predict). All other rows are kept."
        )
        df = df[df['WordType'].notna()].copy()

    df['label'] = np.where(df['WordType'] == 'CV', 1, 0)

    # Missing values are imputed (median, fit per CV fold) inside modeling pipelines.
    # Some missingness is structural: PN_* prenazal measures are often absent for plain
    # 'V' words (no nasal consonant to measure). Prefer NO_PN_FEATURE_COLS for a
    # sensitivity check that avoids imputing structurally absent features.
    n_missing_any = df[CLEAN_FEATURE_COLS].isna().any(axis=1).sum()
    if n_missing_any:
        print(
            f"Note: {n_missing_any}/{len(df)} rows ({_pct(n_missing_any / len(df))}) have at least one "
            f"missing feature value; these are kept and imputed rather than dropped."
        )

    if 'Speaker' not in df.columns:
        speaker_col = _resolve_col(df, SPEAKER_COL_CANDIDATES, 'speaker-ID')
        df['Speaker'] = df[speaker_col]

    if 'POA' in df.columns:
        _init_cons_map = {
            ('CV', 'B'): 'm', ('CV', 'A'): 'n',
            ('PNV', 'B'): 'b', ('PNV', 'A'): 'd',
            ('V', 'B'): 'p', ('V', 'A'): 't',
        }
        df['InitCons'] = [
            _init_cons_map.get((wt, poa), np.nan)
            for wt, poa in zip(df['WordType'], df['POA'])
        ]

    if 'PRODUCTION' in df.columns and 'RAND' in df.columns:
        # Keep true missing separate from the string "ok".
        prod_raw = df['PRODUCTION']
        is_missing = prod_raw.isna() | (prod_raw.astype(str).str.strip() == '')
        prod = prod_raw.astype(str).str.strip()
        is_match = (~is_missing) & prod.isin(['ok'])
        # Missing PRODUCTION: leave as NA rather than pretending it matched the target.
        df['ActualProduction'] = np.where(is_match, df['RAND'], np.where(is_missing, np.nan, df['PRODUCTION']))

    sub_col = 'Substitutions.  AO notations in green.'
    if sub_col in df.columns:
        note = df[sub_col].fillna('').astype(str).str.lower()
        is_initial_note = note.str.contains('init') | note.str.contains(r'^int ')
        produced_oral = is_initial_note & note.str.contains('nasal to oral')
        produced_nasal = is_initial_note & (
            note.str.contains('oral to nasal') | note.str.contains('oral to prenas')
        )

        coded_manner = np.where(df['WordType'] == 'CV', 'nasal', 'oral')
        actual_manner = coded_manner.copy()
        actual_manner[produced_oral.values] = 'oral'
        actual_manner[produced_nasal.values] = 'nasal'
        df['ActualManner'] = actual_manner
        df['InitialSubstitution'] = np.where(produced_oral | produced_nasal, note, '')

    return df


def build_models():
    def make(clf):
        return Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', clf),
        ])

    return {
        'Logistic Regression': make(LogisticRegression(max_iter=5000, class_weight='balanced')),
        'Random Forest': make(RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight='balanced', random_state=0)),
        'Gradient Boosting': make(GradientBoostingClassifier(random_state=0)),
        'SVM (RBF, default)': make(SVC(probability=True, class_weight='balanced', random_state=0)),
        'k-Nearest Neighbors': make(KNeighborsClassifier(n_neighbors=15)),
        'Naive Bayes': make(GaussianNB()),
    }


def run_model_comparison(df, feature_cols=None):
    feature_cols = feature_cols or CLEAN_FEATURE_COLS
    X = df[feature_cols]
    y = df['label'].values
    groups = df['Speaker'].values
    n_splits = _n_group_splits(groups)
    gkf = GroupKFold(n_splits=n_splits)

    rows, probs_by_model = [], {}
    for name, pipe in build_models().items():
        # One CV pass for probabilities; derive hard predictions from them.
        probs = cross_val_predict(pipe, X, y, cv=gkf, groups=groups, method='predict_proba')[:, 1]
        preds = cross_val_predict(pipe, X, y, cv=gkf, groups=groups, method='predict')
        probs_by_model[name] = probs
        rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y, preds),
            'AUC': roc_auc_score(y, probs),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1': f1_score(y, preds, zero_division=0),
        })

    results = pd.DataFrame(rows).sort_values('AUC', ascending=False).reset_index(drop=True)
    return add_percentage_columns(results), probs_by_model, y


def _select_features_on_train(X_train, y_train, groups_train, min_features=1):
    """RFECV fitted only on a training fold (no test-fold leakage)."""
    n_splits = _n_group_splits(groups_train)
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('svc', SVC(kernel='linear', class_weight='balanced', random_state=0)),
    ])
    rfecv = RFECV(
        estimator=pipe,
        step=1,
        cv=GroupKFold(n_splits=n_splits),
        scoring='roc_auc',
        min_features_to_select=min_features,
        importance_getter='named_steps.svc.coef_',
        n_jobs=1,
    )
    rfecv.fit(X_train, y_train, groups=groups_train)
    selected = list(X_train.columns[rfecv.support_])
    return selected


def tune_svm(df, feature_cols=None, use_feature_selection=False):
    """Nested GroupKFold: inner grid search (and optional RFECV) never sees outer test speakers."""
    feature_cols = feature_cols or CLEAN_FEATURE_COLS
    X = df[feature_cols]
    y = df['label'].values
    groups = df['Speaker'].values

    n_splits = _n_group_splits(groups)
    outer_cv = GroupKFold(n_splits=n_splits)

    preds = np.empty(len(y), dtype=int)
    probs = np.empty(len(y), dtype=float)
    fold_importances = []
    fold_params = []
    fold_selected = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        cols = feature_cols
        if use_feature_selection:
            cols = _select_features_on_train(X_train, y_train, groups_train)
            fold_selected.append(cols)
            print(f"  Outer fold {fold_i}: RFECV kept {len(cols)} features: {cols}")

        inner_splits = _n_group_splits(groups_train)
        search = GridSearchCV(
            _make_svm_pipe(),
            SVM_PARAM_GRID,
            scoring='roc_auc',
            cv=GroupKFold(n_splits=inner_splits),
            n_jobs=1,
        )
        search.fit(X_train[cols], y_train, groups=groups_train)
        best_pipe = search.best_estimator_
        fold_params.append(search.best_params_)

        preds[test_idx] = best_pipe.predict(X_test[cols])
        probs[test_idx] = best_pipe.predict_proba(X_test[cols])[:, 1]

        # Importance on the held-out fold only (not the training data).
        r = permutation_importance(
            best_pipe, X_test[cols], y_test,
            n_repeats=20, random_state=0, scoring='roc_auc',
        )
        fold_importances.append(pd.Series(r.importances_mean, index=cols))

    metrics = {
        'Accuracy': accuracy_score(y, preds),
        'AUC': roc_auc_score(y, probs),
        'Precision': precision_score(y, preds, zero_division=0),
        'Recall': recall_score(y, preds, zero_division=0),
        'F1': f1_score(y, preds, zero_division=0),
    }
    cm = confusion_matrix(y, preds)

    # Average permutation importance across folds (union of features; missing fold -> NaN then mean).
    importance = (
        pd.concat(fold_importances, axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
    )
    importance_kind = 'Nested CV permutation importance (mean drop in AUC on held-out folds)'

    # Final model on all data (for support-vector counts / optional deployment only — not for metrics).
    final_cols = feature_cols
    if use_feature_selection and fold_selected:
        # Features kept in a majority of outer folds.
        counts = Counter(c for cols in fold_selected for c in cols)
        final_cols = [c for c, n in counts.items() if n >= (n_splits / 2)]
        if not final_cols:
            final_cols = feature_cols

    final_search = GridSearchCV(
        _make_svm_pipe(),
        SVM_PARAM_GRID,
        scoring='roc_auc',
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=1,
    )
    final_search.fit(X[final_cols], y, groups=groups)
    best_pipe = final_search.best_estimator_
    svc = best_pipe.named_steps['svc']

    # Most common inner-fold params (for reporting); metrics above are nested / unbiased.
    param_counts = pd.Series([str(p) for p in fold_params]).value_counts()
    modal_str = param_counts.index[0]
    most_common_params = next(p for p in fold_params if str(p) == modal_str)

    return {
        'best_params': most_common_params,
        'final_fit_params': final_search.best_params_,
        'metrics': metrics,
        'confusion_matrix': cm,
        'preds': preds,
        'probs': probs,
        'y': y,
        'importance': importance,
        'importance_kind': importance_kind,
        'n_support': svc.n_support_,
        'feature_cols_used': final_cols,
        'fold_selected_features': fold_selected,
        'n_splits': n_splits,
    }


def wordtype_breakdown(df, preds, y, wordtype_col='WordType'):
    """Out-of-fold accuracy by original WordType (CV vs PNV vs V)."""
    correct = (np.asarray(preds) == np.asarray(y))
    tmp = pd.DataFrame({wordtype_col: df[wordtype_col].values, 'correct': correct})
    out = (
        tmp.groupby(wordtype_col)
        .agg(N=('correct', 'size'), Accuracy=('correct', 'mean'))
        .reset_index()
        .sort_values('Accuracy', ascending=False)
        .reset_index(drop=True)
    )
    out['Accuracy %'] = out['Accuracy'].map(_pct)
    out['Correct'] = (out['Accuracy'] * out['N']).round().astype(int)
    return out


def plot_wordtype_accuracy(wt_acc, fig_dir=None, title_suffix=''):
    fig_dir = fig_dir or FIG_DIR
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette('viridis', len(wt_acc))
    bars = ax.bar(wt_acc['WordType'], wt_acc['Accuracy'], color=colors)
    for bar, n, acc in zip(bars, wt_acc['N'], wt_acc['Accuracy']):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{_pct(acc)}\nn={n}', ha='center', fontsize=10,
        )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel('Out-of-fold accuracy (%)')
    ax.set_title(f'Tuned SVM accuracy by WordType{title_suffix}')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/7_wordtype_accuracy.png', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def summarize_feature_dominance(importance, focal_feature='VAvg3'):
    """Rank features by permutation importance and quantify dominance share.

    Dominance share = feature's positive importance / sum of all positive importances.
    Negative importances (hurting the score when shuffled) get share 0.
    """
    importance = pd.Series(importance).sort_values(ascending=False)
    positive = importance.clip(lower=0)
    total_pos = float(positive.sum())
    rows = []
    for rank, (feat, imp) in enumerate(importance.items(), start=1):
        share = (float(positive[feat]) / total_pos) if total_pos > 0 else 0.0
        rows.append({
            'Rank': rank,
            'Feature': feat,
            'Permutation importance': float(imp),
            'Dominance share': share,
            'Dominance share %': _pct(share),
            'Is top': 'Yes' if rank == 1 else '',
            f'Is {focal_feature}': 'Yes' if feat == focal_feature else '',
        })
    table = pd.DataFrame(rows)
    top_feature = table.iloc[0]['Feature'] if len(table) else None
    top_share = float(table.iloc[0]['Dominance share']) if len(table) else np.nan
    focal_rows = table[table['Feature'] == focal_feature]
    if len(focal_rows):
        focal_rank = int(focal_rows.iloc[0]['Rank'])
        focal_share = float(focal_rows.iloc[0]['Dominance share'])
        focal_imp = float(focal_rows.iloc[0]['Permutation importance'])
    else:
        focal_rank, focal_share, focal_imp = None, np.nan, np.nan

    return {
        'table': table,
        'top_feature': top_feature,
        'top_share': top_share,
        'focal_feature': focal_feature,
        'focal_rank': focal_rank,
        'focal_share': focal_share,
        'focal_importance': focal_imp,
        'focal_is_dominant': top_feature == focal_feature,
    }


def print_feature_dominance(dominance, title='Feature dominance'):
    """Print ranked importance + whether the focal feature leads."""
    print(f"\n{title}")
    print(dominance['table'].to_string(index=False))
    top = dominance['top_feature']
    top_share = dominance['top_share']
    focal = dominance['focal_feature']
    print(
        f"\nTop feature: {top} "
        f"(dominance share {_pct(top_share)} of positive importance)"
    )
    if dominance['focal_rank'] is None:
        print(f"{focal}: not present in this feature set.")
    elif dominance['focal_is_dominant']:
        print(
            f"{focal} IS dominant for this task "
            f"(rank #1, share {_pct(dominance['focal_share'])})."
        )
    else:
        print(
            f"{focal} is NOT dominant for this task "
            f"(rank #{dominance['focal_rank']}, share {_pct(dominance['focal_share'])}; "
            f"leader is {top})."
        )
    return dominance


def tune_multiclass_svm(df, target_col, feature_cols=None, class_labels=None,
                        compute_importance=True, focal_feature='VAvg3'):
    """Nested GroupKFold multiclass SVM (hyperparameters chosen inside each outer fold).

    When compute_importance=True (default), also estimates held-out permutation
    importance and a dominance summary (same logic used for binary oral vs nasal).
    """
    feature_cols = feature_cols or CLEAN_FEATURE_COLS
    X = df[feature_cols]
    y = df[target_col].astype(str).values
    groups = df['Speaker'].values
    class_labels = class_labels or sorted(pd.unique(y))

    n_splits = _n_group_splits(groups)
    outer_cv = GroupKFold(n_splits=n_splits)

    preds = np.empty(len(y), dtype=object)
    fold_params = []
    fold_importances = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        inner_splits = _n_group_splits(groups_train)
        search = GridSearchCV(
            _make_svm_pipe(),
            MULTICLASS_PARAM_GRID,
            scoring='accuracy',
            cv=GroupKFold(n_splits=inner_splits),
            n_jobs=1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        best_pipe = search.best_estimator_
        fold_params.append(search.best_params_)
        preds[test_idx] = best_pipe.predict(X_test)

        if compute_importance:
            r = permutation_importance(
                best_pipe, X_test, y_test,
                n_repeats=20, random_state=0, scoring='accuracy',
            )
            fold_importances.append(pd.Series(r.importances_mean, index=feature_cols))
            print(
                f"  Outer fold {fold_i}: top feature = "
                f"{fold_importances[-1].idxmax()} "
                f"(importance={fold_importances[-1].max():.4f})"
            )

    accuracy = accuracy_score(y, preds)
    f1_macro = f1_score(y, preds, average='macro', zero_division=0)
    cm = confusion_matrix(y, preds, labels=class_labels)
    report = classification_report(
        y, preds, labels=class_labels, output_dict=True, zero_division=0,
    )
    per_class = pd.DataFrame(report).T.loc[list(class_labels), ['precision', 'recall', 'f1-score', 'support']]
    per_class = add_percentage_columns(per_class, metric_cols=['precision', 'recall', 'f1-score'])

    majority_baseline = pd.Series(y).value_counts(normalize=True).max()
    chance_baseline = 1.0 / len(class_labels)

    param_counts = pd.Series([str(p) for p in fold_params]).value_counts()
    most_common_params = fold_params[0]
    for p in fold_params:
        if str(p) == param_counts.index[0]:
            most_common_params = p
            break

    importance = None
    importance_kind = None
    dominance = None
    if compute_importance and fold_importances:
        importance = (
            pd.concat(fold_importances, axis=1)
            .mean(axis=1)
            .sort_values(ascending=False)
        )
        importance_kind = (
            'Nested CV permutation importance '
            '(mean drop in accuracy on held-out folds)'
        )
        dominance = summarize_feature_dominance(importance, focal_feature=focal_feature)

    return {
        'target_col': target_col,
        'best_params': most_common_params,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'majority_baseline': majority_baseline,
        'chance_baseline': chance_baseline,
        'confusion_matrix': cm,
        'class_labels': class_labels,
        'per_class': per_class,
        'preds': preds,
        'y': y,
        'importance': importance,
        'importance_kind': importance_kind,
        'dominance': dominance,
        'n_splits': n_splits,
    }


def plot_multiclass_confusion_matrix(cm, class_labels, title, filename, fig_dir=None):
    fig_dir = fig_dir or FIG_DIR
    summary = summarize_confusion_matrix(cm, labels=class_labels)
    fig, axes = plt.subplots(1, 2, figsize=(1.2 * len(class_labels) + 5, 1.1 * len(class_labels) + 2))

    sns.heatmap(
        summary['counts'], annot=True, fmt='d', cmap='viridis',
        xticklabels=class_labels, yticklabels=class_labels,
        ax=axes[0], cbar=False, annot_kws={'size': 11},
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'{title}\n(counts)')

    row_pct = np.divide(
        summary['counts'],
        summary['counts'].sum(axis=1, keepdims=True),
        out=np.zeros_like(summary['counts'], dtype=float),
        where=summary['counts'].sum(axis=1, keepdims=True) != 0,
    )
    sns.heatmap(
        row_pct, annot=True, fmt='.1%', cmap='viridis',
        xticklabels=class_labels, yticklabels=class_labels,
        ax=axes[1], cbar=False, annot_kws={'size': 11},
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Row % (% of each actual class)')

    fig.tight_layout()
    fig.savefig(f'{fig_dir}/{filename}', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)
    return summary


def run_multiclass_analysis(df, targets=('WordType', 'InitCons', 'RAND'), feature_cols=None,
                            fig_dir=None, title_suffix='', focal_feature='VAvg3',
                            importance_targets=('WordType',)):
    """Run multiclass backwards prediction. For targets in importance_targets
    (default: WordType), also compute nested-CV feature dominance, matching the
    binary oral-vs-nasal analysis where VAvg3 dominance was reported."""
    fig_dir = fig_dir or FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)
    feature_cols = feature_cols or CLEAN_FEATURE_COLS

    results = {}
    summary_rows = []
    file_prefix = {'WordType': '8_wordtype', 'InitCons': '9_initcons', 'RAND': '10_fullword'}

    for target in targets:
        if target not in df.columns:
            print(f"Skipping '{target}': not found in dataframe.")
            continue
        print(f"\n--- Predicting {target} from acoustic features ---")
        if target == 'RAND':
            word_list = sorted(pd.Series(df[target]).dropna().astype(str).unique().tolist())
            word_table = pd.DataFrame({
                'Word #': range(1, len(word_list) + 1),
                'Predicted word': word_list,
            })
            print(f"Words included in the {target} task ({len(word_list)} total):")
            print(word_table.to_string(index=False))
            word_table.to_csv(os.path.join(fig_dir, '10_fullword_class_list.csv'), index=False)

        want_importance = target in importance_targets
        res = tune_multiclass_svm(
            df, target, feature_cols=feature_cols,
            compute_importance=want_importance,
            focal_feature=focal_feature,
        )
        results[target] = res
        print("Best params (modal across outer folds):", res['best_params'])
        print(
            f"Accuracy: {res['accuracy']:.3f} ({_pct(res['accuracy'])})  "
            f"(majority-class baseline: {res['majority_baseline']:.3f} ({_pct(res['majority_baseline'])}), "
            f"chance: {res['chance_baseline']:.3f} ({_pct(res['chance_baseline'])}))"
        )
        print(f"Macro F1: {res['f1_macro']:.3f} ({_pct(res['f1_macro'])})")
        print("Per-class performance:")
        print(res['per_class'].to_string())
        print_confusion_matrix_summary(
            res['confusion_matrix'], labels=res['class_labels'],
            title=f'Confusion matrix percentages ({target})',
        )

        prefix = file_prefix.get(target, target.lower())
        plot_multiclass_confusion_matrix(
            res['confusion_matrix'], res['class_labels'],
            f'Predicting {target} from acoustics{title_suffix}',
            f'{prefix}_confusion_matrix.png', fig_dir=fig_dir,
        )

        if res.get('dominance') is not None:
            print_feature_dominance(
                res['dominance'],
                title=(
                    f'Feature dominance for {target} classification '
                    f'(nested CV permutation importance)'
                ),
            )
            dom_path = os.path.join(fig_dir, f'{prefix}_feature_dominance.csv')
            res['dominance']['table'].to_csv(dom_path, index=False)
            print(f"Saved: {dom_path}")

            plot_feature_importance(
                res['importance'], res['importance_kind'],
                f'{prefix}_feature_importance.png', fig_dir=fig_dir,
                title=f'{target}: SVM feature importance{title_suffix}',
            )

            if target == 'WordType' and target in df.columns:
                top_feats = (
                    res['importance'].abs()
                    .sort_values(ascending=False)
                    .index[:3]
                    .tolist()
                )
                top_feats = [f for f in top_feats if f in df.columns]
                if top_feats:
                    plot_wordtype_feature_distributions(
                        df, top_feats, fig_dir=fig_dir, title_suffix=title_suffix,
                    )

        summary_rows.append({
            'Target': target, 'N classes': len(res['class_labels']),
            'Accuracy': res['accuracy'], 'Macro F1': res['f1_macro'],
            'Majority baseline': res['majority_baseline'],
            'Chance baseline': res['chance_baseline'],
            'Top feature': (
                res['dominance']['top_feature'] if res.get('dominance') else ''
            ),
            f'{focal_feature} rank': (
                res['dominance']['focal_rank'] if res.get('dominance') else ''
            ),
            f'{focal_feature} dominance share': (
                res['dominance']['focal_share'] if res.get('dominance') else np.nan
            ),
            f'{focal_feature} is dominant': (
                bool(res['dominance']['focal_is_dominant'])
                if res.get('dominance') else ''
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary = add_percentage_columns(summary)
    print("\n=== Summary: backwards prediction accuracy by target ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(fig_dir, 'multiclass_summary_percentages.csv'), index=False)

    return results, summary


def plot_wordtype_feature_distributions(df, top_features, fig_dir=None, title_suffix=''):
    """Violin plots of top features by the 3-way WordType (CV / PNV / V)."""
    fig_dir = fig_dir or FIG_DIR
    fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 5))
    if len(top_features) == 1:
        axes = [axes]
    order = [c for c in ['CV', 'PNV', 'V'] if c in set(df['WordType'].astype(str))]
    for ax, feat in zip(axes, top_features):
        sns.violinplot(
            data=df, x='WordType', y=feat, order=order, ax=ax,
            palette='viridis',
        )
        ax.set_xlabel('')
        ax.set_title(feat)
    fig.suptitle(
        f'Distribution of top features by WordType (CV / PNV / V){title_suffix}'
    )
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/8_wordtype_feature_distributions.png', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def build_stimulus_production_table(df, multiclass_results):
    cols = ['Speaker']
    if 'Group_1' in df.columns:
        cols.append('Group_1')
    cols += ['RAND', 'WordType', 'InitCons', 'ActualProduction']
    if 'ActualManner' in df.columns:
        cols += ['ActualManner', 'InitialSubstitution']
    table = df[cols].reset_index(drop=True).copy()

    for target, res in multiclass_results.items():
        table[f'{target}_predicted'] = res['preds']
        table[f'{target}_correct'] = (np.asarray(res['preds']) == np.asarray(res['y']))

    return table


def substitution_tracking_check(table):
    if 'ActualManner' not in table.columns or 'WordType_predicted' not in table.columns:
        raise ValueError(
            "Table needs 'ActualManner' and 'WordType_predicted' columns "
            "(run build_stimulus_production_table with a 'WordType' target)."
        )

    coded_manner = np.where(table['WordType'] == 'CV', 'nasal', 'oral')
    sub_mask = coded_manner != table['ActualManner'].values
    n_sub = sub_mask.sum()
    n_total = len(table)
    print(
        f"\nTokens with a coded initial nasal<->oral substitution: "
        f"{n_sub}/{n_total} ({_pct(n_sub / n_total if n_total else 0)})"
    )
    if n_sub == 0:
        print("(none found -- nothing to check)")
        return None

    pred_manner = np.where(table['WordType_predicted'] == 'CV', 'nasal', 'oral')
    matches_target = (pred_manner[sub_mask] == coded_manner[sub_mask]).mean()
    matches_actual = (pred_manner[sub_mask] == table['ActualManner'].values[sub_mask]).mean()

    print("Of these, the model's WordType prediction matched:")
    print(f"  the CODED/intended target manner: {matches_target:.1%} ({int(round(matches_target * n_sub))}/{n_sub})")
    print(f"  the CHILD'S ACTUAL produced manner: {matches_actual:.1%} ({int(round(matches_actual * n_sub))}/{n_sub})")

    detail = table.loc[sub_mask, [
        'Speaker', 'RAND', 'WordType', 'ActualProduction',
        'ActualManner', 'WordType_predicted', 'InitialSubstitution',
    ]]
    return detail


def plot_model_comparison(results, fig_dir=None, title_suffix=''):
    fig_dir = fig_dir or FIG_DIR
    fig, ax = plt.subplots(figsize=(10, 6))
    melted = results.melt(
        id_vars='Model', value_vars=['Accuracy', 'AUC', 'F1'],
        var_name='Metric', value_name='Score',
    )
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', ax=ax, palette='viridis')
    ymin = max(0.0, float(melted['Score'].min()) - 0.05)
    ax.set_ylim(ymin, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(
        f'Model comparison: Accuracy / AUC / F1{title_suffix}\n'
        f'(GroupKFold by Speaker, clean feature set)'
    )
    ax.set_xlabel('')
    plt.xticks(rotation=25, ha='right')
    ax.legend(title='', loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/1_model_comparison_bars.png', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_roc_curves(probs_by_model, y, svm_probs=None, svm_label='Tuned SVM (nested CV)',
                    fig_dir=None, title_suffix=''):
    fig_dir = fig_dir or FIG_DIR
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette('viridis', len(probs_by_model) + (1 if svm_probs is not None else 0))
    for (name, probs), color in zip(probs_by_model.items(), colors):
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        ax.plot(fpr, tpr, label=f'{name} (AUC={_pct(auc)})', color=color, linewidth=2)
    if svm_probs is not None:
        fpr, tpr, _ = roc_curve(y, svm_probs)
        auc = roc_auc_score(y, svm_probs)
        ax.plot(fpr, tpr, label=f'{svm_label} (AUC={_pct(auc)})', color='crimson',
                linewidth=3, linestyle='--')
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC curves: all models{title_suffix}')
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/2_roc_curves.png', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(cm, title, filename, fig_dir=None, labels=None):
    fig_dir = fig_dir or FIG_DIR
    labels = labels or ['Actual oral', 'Actual nasal']
    pred_labels = ['Predicted oral', 'Predicted nasal'] if len(labels) == 2 else [f'Pred {l}' for l in labels]
    summary = summarize_confusion_matrix(cm, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        summary['counts'], annot=True, fmt='d', cmap='viridis',
        xticklabels=pred_labels, yticklabels=labels,
        ax=axes[0], cbar=False, annot_kws={'size': 14},
    )
    axes[0].set_title(f'{title}\n(counts)')

    row_pct = np.divide(
        summary['counts'],
        summary['counts'].sum(axis=1, keepdims=True),
        out=np.zeros_like(summary['counts'], dtype=float),
        where=summary['counts'].sum(axis=1, keepdims=True) != 0,
    )
    sns.heatmap(
        row_pct, annot=True, fmt='.1%', cmap='viridis',
        xticklabels=pred_labels, yticklabels=labels,
        ax=axes[1], cbar=False, annot_kws={'size': 14},
    )
    axes[1].set_title('Row % (% of each actual class)')

    fig.tight_layout()
    fig.savefig(f'{fig_dir}/{filename}', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)
    return summary


def plot_feature_importance(importance, kind, filename, fig_dir=None, title='SVM feature importance'):
    fig_dir = fig_dir or FIG_DIR
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['crimson' if v < 0 else PALETTE[3] for v in importance.values]
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_xlabel(kind)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/{filename}', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_feature_distributions(df, top_features, fig_dir=None, title_suffix=''):
    fig_dir = fig_dir or FIG_DIR
    fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 5))
    if len(top_features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, top_features):
        sns.violinplot(
            data=df, x='label', y=feat, hue='label', ax=ax,
            palette=['#3b528b', '#5ec962'], legend=False,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Oral', 'Nasal'])
        ax.set_xlabel('')
        ax.set_title(feat)
    fig.suptitle(f'Distribution of top acoustic features by category{title_suffix}')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/5_feature_distributions.png', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def plot_support_vectors(n_support, fig_dir=None):
    fig_dir = fig_dir or FIG_DIR
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(['Oral (class 0)', 'Nasal (class 1)'], n_support, color=['#3b528b', '#5ec962'])
    ax.set_ylabel('Number of support vectors')
    ax.set_title('SVM support vectors by class (final refit; not used for metrics)')
    for i, v in enumerate(n_support):
        ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/6_support_vectors.png', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def select_features(df, feature_cols=None, min_features=1):
    """Exploratory RFECV on the full dataset (for plots only — not for final metrics)."""
    feature_cols = feature_cols or CLEAN_FEATURE_COLS
    X = df[feature_cols]
    y = df['label'].values
    groups = df['Speaker'].values
    n_splits = _n_group_splits(groups)
    gkf = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('svc', SVC(kernel='linear', class_weight='balanced', random_state=0)),
    ])

    rfecv = RFECV(
        estimator=pipe,
        step=1,
        cv=gkf,
        scoring='roc_auc',
        min_features_to_select=min_features,
        importance_getter='named_steps.svc.coef_',
        n_jobs=1,
    )
    rfecv.fit(X, y, groups=groups)

    selected = list(X.columns[rfecv.support_])
    dropped = [c for c in feature_cols if c not in selected]

    print(f"[Exploratory] Started with {len(feature_cols)} features, RFECV selected {len(selected)} ({_pct(len(selected) / len(feature_cols))}).")
    print(f"Kept:    {selected}")
    print(f"Dropped: {dropped}")
    print("(Final metrics nest feature selection inside each outer CV fold when enabled.)")

    return rfecv, selected, dropped


def rfecv_trial_table(rfecv, feature_cols):
    """Build a table of RFECV trials with feature names inferred from the final RFECV ranking."""
    scores = np.asarray(rfecv.cv_results_['mean_test_score'])
    n_trials = len(scores)
    n_kept_values = list(range(rfecv.min_features_to_select, rfecv.min_features_to_select + n_trials))
    best_idx = int(np.argmax(scores))
    ranking = pd.Series(rfecv.ranking_, index=feature_cols)
    ranked_features = ranking.sort_values(kind='stable').index.tolist()

    rows = []
    for trial_idx, (n_kept, score) in enumerate(zip(n_kept_values, scores), start=1):
        kept_features = ranked_features[:n_kept]
        dropped_features = ranked_features[n_kept:]
        rows.append({
            'Trial': trial_idx,
            'Features tried': n_kept,
            'Features dropped': len(feature_cols) - n_kept,
            'Kept feature names': ', '.join(kept_features),
            'Dropped feature names': ', '.join(dropped_features),
            'Mean CV AUC': score,
            'Mean CV AUC %': _pct(score),
            'Worked best': 'Yes' if (trial_idx - 1) == best_idx else '',
        })

    return pd.DataFrame(rows)


def print_rfecv_trial_table(rfecv, feature_cols, fig_dir=None):
    """Print and save the RFECV trial summary table."""
    table = rfecv_trial_table(rfecv, feature_cols)
    print("\nRFECV trial table (feature names inferred from final RFECV ranking):")
    print(table.to_string(index=False))
    if fig_dir is not None:
        out_path = os.path.join(fig_dir, '0_rfecv_trial_table.csv')
        table.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    return table


def plot_rfecv_curve(rfecv, fig_dir=None, title_suffix=''):
    fig_dir = fig_dir or FIG_DIR
    scores = rfecv.cv_results_['mean_test_score']
    n_features = range(1, len(scores) + 1)
    best_n = np.argmax(scores) + 1

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(n_features, scores, marker='o', color='#2E7D6E')
    ax.axvline(best_n, color='crimson', linestyle='--', label=f'Best: {best_n} features')
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Cross-validated AUC')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f'RFECV (exploratory): performance vs. features kept{title_suffix}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{fig_dir}/0_rfecv_curve.png', dpi=150)
    if _HAS_IPYTHON:
        display(fig)
    plt.show()
    plt.close(fig)


def compare_full_vs_reduced(df, selected_cols, full_cols=None):
    """Cross-validated comparison of full vs reduced feature sets (fixed subsets; no tuning)."""
    full_cols = full_cols or CLEAN_FEATURE_COLS
    y = df['label'].values
    groups = df['Speaker'].values
    n_splits = _n_group_splits(groups)
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    for label, cols in [('Full', full_cols), ('Reduced', selected_cols)]:
        X = df[cols]
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('svc', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=0)),
        ])
        preds = cross_val_predict(pipe, X, y, cv=gkf, groups=groups, method='predict')
        probs = cross_val_predict(pipe, X, y, cv=gkf, groups=groups, method='predict_proba')[:, 1]
        rows.append({
            'Feature set': f'{label} ({len(cols)} features)',
            'Accuracy': accuracy_score(y, preds),
            'AUC': roc_auc_score(y, probs),
            'F1': f1_score(y, preds, zero_division=0),
        })

    comparison = add_percentage_columns(pd.DataFrame(rows))
    print_metric_table(comparison, title='Full vs reduced feature set (GroupKFold):')
    return comparison


def pn_sensitivity_check(df, use_feature_selection=False):
    """Compare nested SVM with vs without structurally missing PN_* features."""
    print("\n### Sensitivity: with PN_* features vs without (nested CV) ###")
    with_pn = tune_svm(df, feature_cols=CLEAN_FEATURE_COLS,
                       use_feature_selection=use_feature_selection)
    no_pn = tune_svm(df, feature_cols=NO_PN_FEATURE_COLS,
                     use_feature_selection=use_feature_selection)

    summary = add_percentage_columns(pd.DataFrame([
        {'Feature set': f'With PN_* ({len(CLEAN_FEATURE_COLS)})', **with_pn['metrics']},
        {'Feature set': f'Without PN_* ({len(NO_PN_FEATURE_COLS)})', **no_pn['metrics']},
    ]))
    print_metric_table(summary, title='PN_* sensitivity (nested CV):')
    summary.to_csv(os.path.join(FIG_DIR, 'pn_sensitivity_percentages.csv'), index=False)
    print(
        "If AUC drops a lot without PN_*, imputed prenazal measures may be carrying "
        "signal that is partly about missingness rather than acoustics alone."
    )
    return summary, with_pn, no_pn


SUBSTITUTION_COL = 'Substitutions.  AO notations in green.'
WORDTYPE_ORDER = ['CV', 'PNV', 'V']


def _wordtype_levels(df):
    present = set(df['WordType'].astype(str))
    return [w for w in WORDTYPE_ORDER if w in present] + sorted(present - set(WORDTYPE_ORDER))


def pn_missingness_report(df, fig_dir):
    """Describe where PN_* values are missing vs observed, by WordType. Report only."""
    os.makedirs(fig_dir, exist_ok=True)
    pn_cols = [c for c in PN_FEATURE_COLS if c in df.columns]
    levels = _wordtype_levels(df)
    wt = df['WordType'].astype(str)
    missing = df[pn_cols].isna()

    print("\n### PN missingness report ###")
    print(f"N = {len(df)} rows; PN columns: {pn_cols}")

    # a. Missing count / % by WordType, per PN column
    rows = []
    for col in pn_cols:
        for level in levels + ['All']:
            mask = np.ones(len(df), dtype=bool) if level == 'All' else (wt == level).values
            n = int(mask.sum())
            n_miss = int(missing.loc[mask, col].sum())
            rows.append({
                'PN column': col, 'WordType': level, 'N tokens': n,
                'N missing': n_miss, '% missing': _pct(n_miss / n) if n else 'NA',
            })
    by_wt = pd.DataFrame(rows)
    print("\n(a) Missing PN values by WordType:")
    print(by_wt.to_string(index=False))
    by_wt.to_csv(os.path.join(fig_dir, 'pn_missingness_by_wordtype.csv'), index=False)

    # b. Are the PN columns missing together?
    n_missing_per_row = missing.sum(axis=1)
    co = (
        pd.crosstab(wt, n_missing_per_row)
        .reindex(index=levels, columns=range(len(pn_cols) + 1), fill_value=0)
    )
    co.columns = [f'{k} of {len(pn_cols)} missing' for k in co.columns]
    co.loc['All'] = co.sum()
    partial = (n_missing_per_row > 0) & (n_missing_per_row < len(pn_cols))
    n_partial = int(partial.sum())
    print(f"\n(b) Number of PN columns missing per row, by WordType:")
    print(co.to_string())
    if n_partial == 0:
        print(f"All {len(pn_cols)} PN columns are always missing together (no partial rows).")
    else:
        print(f"NOT always missing together: {n_partial}/{len(df)} rows "
              f"({_pct(n_partial / len(df))}) have 1-{len(pn_cols) - 1} of {len(pn_cols)} missing.")
        patterns = (
            missing.loc[partial]
            .apply(lambda r: ', '.join(c for c in pn_cols if r[c]), axis=1)
            .rename('Missing columns')
        )
        pattern_counts = (
            pd.DataFrame({'WordType': wt[partial], 'Missing columns': patterns})
            .value_counts().rename('N rows').reset_index()
        )
        print("Partial-missingness patterns:")
        print(pattern_counts.to_string(index=False))
    co.to_csv(os.path.join(fig_dir, 'pn_comissingness_by_wordtype.csv'))

    # c. Observed-value range by WordType
    rows = []
    for col in pn_cols:
        for level in levels:
            vals = df.loc[(wt == level).values, col].dropna().astype(float).values
            if len(vals):
                stats = {
                    'Min': vals.min(), 'P5': np.percentile(vals, 5),
                    'Median': np.median(vals), 'P95': np.percentile(vals, 95),
                    'Max': vals.max(),
                }
            else:
                stats = {k: np.nan for k in ['Min', 'P5', 'Median', 'P95', 'Max']}
            rows.append({'PN column': col, 'WordType': level, 'N observed': len(vals), **stats})
    observed_range = pd.DataFrame(rows)
    print("\n(c) Observed (non-missing) PN values by WordType:")
    print(observed_range.to_string(index=False))
    observed_range.to_csv(os.path.join(fig_dir, 'pn_observed_range_by_wordtype.csv'), index=False)

    # d. V tokens with >=1 PN column missing
    v_mask = (wt == 'V').values
    n_v = int(v_mask.sum())
    n_v_missing = int(missing.loc[v_mask].any(axis=1).sum())
    print(f"\n(d) V tokens with >=1 PN column missing: {n_v_missing}/{n_v} "
          f"({_pct(n_v_missing / n_v) if n_v else 'NA'})")

    # e. CV and V tokens that have at least one observed PN value
    has_obs = (~missing).any(axis=1).values
    non_pnv = wt.isin(['CV', 'V']).values
    cols = [c for c in ['Speaker', 'Group_1', 'WordType', 'RAND'] if c in df.columns] + pn_cols
    if SUBSTITUTION_COL in df.columns:
        cols.append(SUBSTITUTION_COL)
    observed_non_pnv = df.loc[has_obs & non_pnv, cols].reset_index(drop=True)
    print(f"\n(e) CV and V tokens with an observed PN value: {len(observed_non_pnv)} rows "
          f"(CV: {int((has_obs & (wt == 'CV').values).sum())}, "
          f"V: {int((has_obs & v_mask).sum())})")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           'display.width', 250):
        print(observed_non_pnv.to_string(index=False))
    observed_non_pnv.to_csv(os.path.join(fig_dir, 'pn_observed_in_nonPNV.csv'), index=False)

    return {
        'by_wordtype': by_wt,
        'comissingness': co,
        'n_partial_rows': n_partial,
        'observed_range': observed_range,
        'v_missing': (n_v_missing, n_v),
        'observed_non_pnv': observed_non_pnv,
    }


def main(source=None, use_feature_selection=True, group_col=None, group_value=None,
         fig_dir=None, title_suffix='', min_group_n=20, run_pn_sensitivity=True,
         feature_cols=None, run_multiclass=True):
    """Run the full pipeline with nested CV for the tuned SVM metrics."""
    if source is None:
        raw_df = read_csv_maybe_ssl(_default_sheet_url())
    else:
        raw_df = source

    df = load_data(raw_df)

    if group_col is not None:
        if group_col not in df.columns:
            raise KeyError(f"'{group_col}' not found. Available columns: {list(df.columns)}")
        df = df[df[group_col] == group_value].copy()
        fig_dir = fig_dir or os.path.join(FIG_DIR, str(group_value))
        title_suffix = title_suffix or f' ({group_value})'
        print(f"=== Group: {group_col} == {group_value!r} ===")
    fig_dir = fig_dir or FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)

    pn_missingness_report(df, fig_dir)

    feature_cols = feature_cols or CLEAN_FEATURE_COLS

    n_speakers = df['Speaker'].nunique()
    n_splits = _n_group_splits(df['Speaker'].values)
    n_missing_any = df[feature_cols].isna().any(axis=1).sum()
    n_nasal = int(df['label'].sum())
    n_oral = int((df['label'] == 0).sum())
    n_total = len(df)
    print(
        f"N = {n_total} rows "
        f"({n_nasal} nasal [{_pct(n_nasal / n_total)}] / "
        f"{n_oral} oral [{_pct(n_oral / n_total)}]), "
        f"{n_speakers} unique speakers, GroupKFold n_splits={n_splits} "
        f"({n_missing_any} rows [{_pct(n_missing_any / n_total)}] have >=1 imputed feature)\n"
    )

    if len(df) < min_group_n:
        print(f"WARNING: only {len(df)} rows (< min_group_n={min_group_n}) -- results may be unstable.\n")
    if n_speakers < 2:
        raise ValueError(f"Need at least 2 speakers for GroupKFold; found {n_speakers}.")

    # Exploratory RFECV only (does not feed into nested metrics unless use_feature_selection
    # nests selection inside each outer fold below).
    if use_feature_selection:
        print("### Step 0: exploratory feature selection (RFECV on full data) ###")
        rfecv, selected, dropped = select_features(df, feature_cols=feature_cols)
        rfecv_trials = print_rfecv_trial_table(rfecv, feature_cols, fig_dir=fig_dir)
        plot_rfecv_curve(rfecv, fig_dir=fig_dir, title_suffix=title_suffix)
        compare_full_vs_reduced(df, selected, full_cols=feature_cols)
        svm_result_rfecv_trials = rfecv_trials
        print()

    print("### Step 1: model comparison (untuned; GroupKFold) ###")
    results, probs_by_model, y = run_model_comparison(df, feature_cols=feature_cols)
    print_metric_table(
        results,
        title='Model comparison:',
        save_path=os.path.join(fig_dir, '1_model_comparison_percentages.csv'),
    )
    plot_model_comparison(results, fig_dir=fig_dir, title_suffix=title_suffix)

    print("\n### Step 2: nested-CV tuned SVM ###")
    svm_result = tune_svm(
        df, feature_cols=feature_cols, use_feature_selection=use_feature_selection,
    )
    print("Modal inner-fold params:", svm_result['best_params'])
    print("Final refit params (not used for metrics):", svm_result['final_fit_params'])
    for line in format_metrics_dict(svm_result['metrics']):
        print(line)
    cm_summary = print_confusion_matrix_summary(
        svm_result['confusion_matrix'],
        labels=['Oral', 'Nasal'],
        title='Nested CV SVM confusion matrix',
    )
    svm_result['confusion_matrix_summary'] = cm_summary
    print(f"\n{svm_result['importance_kind']}:")
    print(svm_result['importance'].to_string())
    binary_dominance = summarize_feature_dominance(svm_result['importance'], focal_feature='VAvg3')
    svm_result['dominance'] = binary_dominance
    print_feature_dominance(
        binary_dominance,
        title='Feature dominance for binary oral vs nasal (nested CV)',
    )
    binary_dominance['table'].to_csv(
        os.path.join(fig_dir, '4_binary_feature_dominance.csv'), index=False,
    )

    wt_acc = wordtype_breakdown(df, svm_result['preds'], svm_result['y'])
    svm_result['wordtype_accuracy'] = wt_acc
    print("\nAccuracy by WordType (nested tuned SVM, out-of-fold):")
    print(wt_acc.to_string(index=False))
    wt_acc.to_csv(os.path.join(fig_dir, '7_wordtype_accuracy_percentages.csv'), index=False)

    metrics_df = add_percentage_columns(pd.DataFrame([svm_result['metrics']]))
    metrics_df.to_csv(os.path.join(fig_dir, '2_svm_metrics_percentages.csv'), index=False)
    if use_feature_selection:
        svm_result['rfecv_trials'] = svm_result_rfecv_trials

    if run_pn_sensitivity and feature_cols == CLEAN_FEATURE_COLS:
        pn_summary, _, _ = pn_sensitivity_check(df, use_feature_selection=False)
        svm_result['pn_sensitivity'] = pn_summary
        pn_summary.to_csv(os.path.join(fig_dir, 'pn_sensitivity_percentages.csv'), index=False)

    print("\n### Step 3: backwards multiclass prediction ###")
    if run_multiclass:
        multiclass_results, multiclass_summary = run_multiclass_analysis(
            df, feature_cols=feature_cols, fig_dir=fig_dir, title_suffix=title_suffix,
        )
    else:
        print("(skipped — run_multiclass=False)")
        multiclass_results, multiclass_summary = {}, None
    svm_result['multiclass_results'] = multiclass_results
    svm_result['multiclass_summary'] = multiclass_summary

    print("\n### Step 4: substitution tracking ###")
    if multiclass_results:
        stimulus_table = build_stimulus_production_table(df, multiclass_results)
        if 'WordType' in multiclass_results:
            substitution_detail = substitution_tracking_check(stimulus_table)
            if substitution_detail is not None:
                substitution_detail.to_csv(
                    os.path.join(fig_dir, 'substitution_tracking_detail.csv'), index=False,
                )
    else:
        print("(skipped — no multiclass targets available)")

    print("\n### Generating visuals ###")
    plot_roc_curves(
        probs_by_model, y, svm_probs=svm_result['probs'],
        fig_dir=fig_dir, title_suffix=title_suffix,
    )
    plot_confusion_matrix(
        svm_result['confusion_matrix'],
        f'Nested CV SVM: Confusion Matrix{title_suffix}',
        '3_svm_confusion_matrix.png', fig_dir=fig_dir,
    )
    plot_feature_importance(
        svm_result['importance'], svm_result['importance_kind'],
        '4_svm_feature_importance.png', fig_dir=fig_dir,
    )
    top_feats = svm_result['importance'].abs().sort_values(ascending=False).index[:3].tolist()
    # Only plot features that exist in df
    top_feats = [f for f in top_feats if f in df.columns][:3]
    if top_feats:
        plot_feature_distributions(df, top_feats, fig_dir=fig_dir, title_suffix=title_suffix)
    plot_support_vectors(svm_result['n_support'], fig_dir=fig_dir)
    plot_wordtype_accuracy(wt_acc, fig_dir=fig_dir, title_suffix=title_suffix)

    print(f"\nAll figures saved to {fig_dir}/")
    return df, results, svm_result


def run_dominance_by_group(source=None, group_col='Group_1',
                           group_labels=None, include_pooled=True,
                           feature_cols=None, focal_feature='VAvg3',
                           use_feature_selection=False, fig_dir=None):
    """Compute nested-CV feature dominance for binary oral/nasal and 3-way WordType
    separately for Hearing (NH), CI, and Pooled.

    Saves per-group tables under fig_dir/<Group>/ and a combined comparison CSV.
    """
    feature_cols = feature_cols or CLEAN_FEATURE_COLS
    fig_dir = fig_dir or FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)

    if source is None:
        raw = read_csv_maybe_ssl(_default_sheet_url())
    else:
        raw = source
    df_all = load_data(raw)

    if group_col not in df_all.columns:
        raise KeyError(f"'{group_col}' not found. Available: {list(df_all.columns)}")

    # Friendly display names: NH -> Hearing
    default_labels = {'NH': 'Hearing', 'CI': 'CI'}
    group_labels = group_labels or default_labels

    slices = []
    if include_pooled:
        slices.append(('Pooled', df_all.copy()))
    for gval in df_all[group_col].dropna().unique():
        label = group_labels.get(gval, str(gval))
        slices.append((label, df_all[df_all[group_col] == gval].copy()))

    print(f"Dominance by group ({group_col}): {[s[0] for s in slices]}\n")

    binary_rows = []
    wordtype_rows = []
    detailed = {}

    for gname, df in slices:
        print("\n" + "=" * 60)
        print(f"=== Dominance: {gname} (N={len(df)}, speakers={df['Speaker'].nunique()}) ===")
        print("=" * 60)
        out_dir = os.path.join(fig_dir, gname)
        os.makedirs(out_dir, exist_ok=True)

        n_speakers = df['Speaker'].nunique()
        if n_speakers < 2:
            print(f"Skipping {gname}: need >=2 speakers for GroupKFold.")
            continue

        # --- Binary oral vs nasal ---
        print(f"\n--- {gname}: binary oral vs nasal ---")
        svm_result = tune_svm(
            df, feature_cols=feature_cols, use_feature_selection=use_feature_selection,
        )
        bin_dom = summarize_feature_dominance(svm_result['importance'], focal_feature=focal_feature)
        print_feature_dominance(
            bin_dom, title=f'Binary dominance ({gname})',
        )
        bin_path = os.path.join(out_dir, '4_binary_feature_dominance.csv')
        bin_dom['table'].to_csv(bin_path, index=False)
        print(f"Saved: {bin_path}")
        plot_feature_importance(
            svm_result['importance'], svm_result['importance_kind'],
            '4_svm_feature_importance.png', fig_dir=out_dir,
            title=f'Binary feature importance ({gname})',
        )

        binary_rows.append({
            'Group': gname,
            'N': len(df),
            'N speakers': n_speakers,
            'Task': 'Binary (oral vs nasal)',
            'Accuracy': svm_result['metrics']['Accuracy'],
            'AUC': svm_result['metrics']['AUC'],
            'Top feature': bin_dom['top_feature'],
            'Top dominance share': bin_dom['top_share'],
            f'{focal_feature} rank': bin_dom['focal_rank'],
            f'{focal_feature} dominance share': bin_dom['focal_share'],
            f'{focal_feature} is dominant': bin_dom['focal_is_dominant'],
        })

        # --- 3-way WordType ---
        print(f"\n--- {gname}: WordType (CV / PNV / V) ---")
        wt_res = tune_multiclass_svm(
            df, 'WordType', feature_cols=feature_cols,
            compute_importance=True, focal_feature=focal_feature,
        )
        wt_dom = wt_res['dominance']
        print(
            f"WordType accuracy: {wt_res['accuracy']:.3f} ({_pct(wt_res['accuracy'])})  "
            f"(chance {_pct(wt_res['chance_baseline'])})"
        )
        print_feature_dominance(
            wt_dom, title=f'WordType dominance ({gname})',
        )
        wt_path = os.path.join(out_dir, '8_wordtype_feature_dominance.csv')
        wt_dom['table'].to_csv(wt_path, index=False)
        print(f"Saved: {wt_path}")
        plot_feature_importance(
            wt_res['importance'], wt_res['importance_kind'],
            '8_wordtype_feature_importance.png', fig_dir=out_dir,
            title=f'WordType feature importance ({gname})',
        )

        wordtype_rows.append({
            'Group': gname,
            'N': len(df),
            'N speakers': n_speakers,
            'Task': 'WordType (CV/PNV/V)',
            'Accuracy': wt_res['accuracy'],
            'Macro F1': wt_res['f1_macro'],
            'Top feature': wt_dom['top_feature'],
            'Top dominance share': wt_dom['top_share'],
            f'{focal_feature} rank': wt_dom['focal_rank'],
            f'{focal_feature} dominance share': wt_dom['focal_share'],
            f'{focal_feature} is dominant': wt_dom['focal_is_dominant'],
        })

        detailed[gname] = {
            'binary': bin_dom,
            'wordtype': wt_dom,
            'binary_metrics': svm_result['metrics'],
            'wordtype_accuracy': wt_res['accuracy'],
        }

    comparison = add_percentage_columns(pd.DataFrame(binary_rows + wordtype_rows))
    # Prefer display order: Pooled, Hearing, CI
    order = {'Pooled': 0, 'Hearing': 1, 'NH': 1, 'CI': 2}
    comparison['_ord'] = comparison['Group'].map(lambda g: order.get(g, 99))
    comparison = (
        comparison.sort_values(['Task', '_ord', 'Group'])
        .drop(columns='_ord')
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("=== Dominance comparison: Hearing / CI / Pooled ===")
    print("=" * 60)
    print(comparison.to_string(index=False))
    out_csv = os.path.join(fig_dir, 'dominance_by_group_summary.csv')
    comparison.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Compact VAvg3-focused pivot
    focal_rows = []
    for task_key, rows in [('Binary', binary_rows), ('WordType', wordtype_rows)]:
        for r in rows:
            focal_rows.append({
                'Group': r['Group'],
                'Task': task_key,
                f'{focal_feature} rank': r[f'{focal_feature} rank'],
                f'{focal_feature} dominance share': r[f'{focal_feature} dominance share'],
                f'{focal_feature} dominance share %': _pct(r[f'{focal_feature} dominance share']),
                'Top feature': r['Top feature'],
                f'{focal_feature} is dominant': r[f'{focal_feature} is dominant'],
            })
    focal_table = pd.DataFrame(focal_rows)
    focal_csv = os.path.join(fig_dir, f'{focal_feature}_dominance_by_group.csv')
    focal_table.to_csv(focal_csv, index=False)
    print(f"Saved: {focal_csv}")
    print(f"\n{focal_feature} dominance by group:")
    print(focal_table.to_string(index=False))

    return detailed, comparison, focal_table


def run_by_group(source, group_col='Group_1', use_feature_selection=True, min_group_n=20,
                 include_pooled=True, run_pn_sensitivity=True):
    df_all = load_data(source)
    if group_col not in df_all.columns:
        raise KeyError(f"'{group_col}' not found. Available columns: {list(df_all.columns)}")

    group_values = [g for g in df_all[group_col].dropna().unique()]
    print(f"Splitting by '{group_col}': found groups {group_values}\n")

    runs = {}
    summary_rows = []

    if include_pooled:
        print("\n" + "=" * 60)
        print("=== Pooled (all groups combined) ===")
        print("=" * 60)
        df, results, svm_result = main(
            source, use_feature_selection=use_feature_selection,
            fig_dir=os.path.join(FIG_DIR, 'Pooled'),
            title_suffix=' (Pooled)', min_group_n=min_group_n,
            run_pn_sensitivity=run_pn_sensitivity,
        )
        runs['Pooled'] = {'df': df, 'results': results, 'svm_result': svm_result}
        summary_rows.append({
            'Group': 'Pooled', 'N': len(df), 'N speakers': df['Speaker'].nunique(),
            **svm_result['metrics'],
            'Top feature': svm_result['importance'].abs().sort_values(ascending=False).index[0],
        })

    for gval in group_values:
        print("\n" + "=" * 60)
        print(f"=== Group: {gval} ===")
        print("=" * 60)
        df, results, svm_result = main(
            source, use_feature_selection=use_feature_selection,
            group_col=group_col, group_value=gval,
            min_group_n=min_group_n,
            run_pn_sensitivity=run_pn_sensitivity,
        )
        runs[gval] = {'df': df, 'results': results, 'svm_result': svm_result}
        summary_rows.append({
            'Group': gval, 'N': len(df), 'N speakers': df['Speaker'].nunique(),
            **svm_result['metrics'],
            'Top feature': svm_result['importance'].abs().sort_values(ascending=False).index[0],
        })

    summary = add_percentage_columns(pd.DataFrame(summary_rows))
    print("\n" + "=" * 60)
    print("=== Summary: nested tuned SVM performance by group ===")
    print("=" * 60)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(FIG_DIR, 'group_svm_summary_percentages.csv'), index=False)

    # Cross-group dominance summary from whatever main() already computed
    dom_rows = []
    for gname, run in runs.items():
        svm = run['svm_result']
        label = {'NH': 'Hearing', 'CI': 'CI'}.get(gname, gname)
        if svm.get('dominance'):
            d = svm['dominance']
            dom_rows.append({
                'Group': label, 'Task': 'Binary',
                'Top feature': d['top_feature'],
                'VAvg3 rank': d['focal_rank'],
                'VAvg3 dominance share': d['focal_share'],
                'VAvg3 is dominant': d['focal_is_dominant'],
            })
        mc = svm.get('multiclass_results') or {}
        if 'WordType' in mc and mc['WordType'].get('dominance'):
            d = mc['WordType']['dominance']
            dom_rows.append({
                'Group': label, 'Task': 'WordType',
                'Top feature': d['top_feature'],
                'VAvg3 rank': d['focal_rank'],
                'VAvg3 dominance share': d['focal_share'],
                'VAvg3 is dominant': d['focal_is_dominant'],
            })
    if dom_rows:
        dom_summary = pd.DataFrame(dom_rows)
        print("\n=== Dominance by group (from full pipeline) ===")
        print(dom_summary.to_string(index=False))
        dom_summary.to_csv(os.path.join(FIG_DIR, 'dominance_by_group_summary.csv'), index=False)

    wt_rows = []
    for gname, run in runs.items():
        wt = run['svm_result']['wordtype_accuracy']
        for _, r in wt.iterrows():
            wt_rows.append({
                'Group': gname, 'WordType': r['WordType'],
                'N': r['N'], 'Correct': r.get('Correct', np.nan),
                'Accuracy': r['Accuracy'], 'Accuracy %': r.get('Accuracy %', _pct(r['Accuracy'])),
            })
    wordtype_summary = pd.DataFrame(wt_rows)
    wordtype_pivot = wordtype_summary.pivot(index='Group', columns='WordType', values='Accuracy')
    wordtype_pivot_pct = wordtype_pivot.map(_pct)
    print("\n" + "=" * 60)
    print("=== Summary: nested SVM accuracy by Group x WordType (proportions) ===")
    print("=" * 60)
    print(wordtype_pivot.to_string())
    print("\n=== Summary: nested SVM accuracy by Group x WordType (%) ===")
    print("=" * 60)
    print(wordtype_pivot_pct.to_string())
    wordtype_summary.to_csv(os.path.join(FIG_DIR, 'group_wordtype_accuracy_percentages.csv'), index=False)
    wordtype_pivot_pct.to_csv(os.path.join(FIG_DIR, 'group_wordtype_accuracy_pivot_percentages.csv'))

    return runs, summary, wordtype_pivot


BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'baseline')


def _write_baseline_tables(runs, out_dir):
    """Write combined baseline tables (one 'Run' column per table) from collected runs."""
    binary, wordtype, folds, importance, pn_sens, support, params = [], [], [], [], [], [], {}
    for run_name, (df, svm) in runs.items():
        m = svm['metrics']
        binary.append({
            'Run': run_name, 'N': len(df), 'N speakers': df['Speaker'].nunique(),
            'Accuracy': m['Accuracy'], 'AUC': m['AUC'], 'F1': m['F1'],
            'Precision': m['Precision'], 'Recall': m['Recall'],
        })
        wordtype.append(svm['wordtype_accuracy'].assign(Run=run_name))
        for i, cols in enumerate(svm.get('fold_selected_features') or [], start=1):
            folds.append({
                'Run': run_name, 'Outer fold': i, 'N selected': len(cols),
                'Selected features': ', '.join(cols),
            })
        importance.append(pd.DataFrame({
            'Run': run_name,
            'Feature': svm['importance'].index,
            'Permutation importance': svm['importance'].values,
        }))
        if svm.get('pn_sensitivity') is not None:
            pn_sens.append(svm['pn_sensitivity'].assign(Run=run_name))
        for cls, n in zip(['Oral (0)', 'Nasal (1)'], svm['n_support']):
            support.append({'Run': run_name, 'Class': cls, 'N support vectors': int(n)})
        params[run_name] = {
            'final_fit_params': svm['final_fit_params'],
            'modal_outer_fold_params': svm['best_params'],
        }

    def _front(frame):
        return frame[['Run'] + [c for c in frame.columns if c != 'Run']]

    pd.DataFrame(binary).to_csv(os.path.join(out_dir, 'binary_metrics.csv'), index=False)
    _front(pd.concat(wordtype, ignore_index=True)).to_csv(
        os.path.join(out_dir, 'wordtype_accuracy.csv'), index=False)
    pd.DataFrame(folds, columns=['Run', 'Outer fold', 'N selected', 'Selected features']).to_csv(
        os.path.join(out_dir, 'rfecv_outer_fold_features.csv'), index=False)
    pd.concat(importance, ignore_index=True).to_csv(
        os.path.join(out_dir, 'permutation_importance.csv'), index=False)
    if pn_sens:
        _front(pd.concat(pn_sens, ignore_index=True)).to_csv(
            os.path.join(out_dir, 'pn_sensitivity.csv'), index=False)
    pd.DataFrame(support).to_csv(os.path.join(out_dir, 'n_support.csv'), index=False)
    with open(os.path.join(out_dir, 'final_fit_params.json'), 'w') as f:
        json.dump(params, f, indent=2, default=str)


def save_baseline(source=None, out_dir=None):
    """Run main() with current settings on the full data and on each Group_1 value,
    and save the reference outputs that later changes must reproduce exactly."""
    out_dir = out_dir or BASELINE_DIR
    os.makedirs(out_dir, exist_ok=True)
    raw = read_csv_maybe_ssl(_default_sheet_url()) if source is None else source
    if not isinstance(raw, pd.DataFrame):
        raw = load_data(raw)

    group_values = sorted(raw['Group_1'].dropna().unique())
    plan = [('full', None)] + [(str(g), g) for g in group_values]

    runs = {}
    for run_name, gval in plan:
        print("\n" + "=" * 60)
        print(f"=== Baseline run: {run_name} ===")
        print("=" * 60)
        if gval is None:
            df, _, svm = main(raw, fig_dir=os.path.join(out_dir, run_name),
                              run_multiclass=False)
        else:
            df, _, svm = main(raw, group_col='Group_1', group_value=gval,
                              fig_dir=os.path.join(out_dir, run_name),
                              run_multiclass=False)
        runs[run_name] = (df, svm)
        _write_baseline_tables(runs, out_dir)
        print(f"Baseline tables updated in {out_dir}/ (runs so far: {list(runs)})")

    return runs


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'baseline':
        save_baseline()
    else:
        # Default: Hearing / CI / Pooled dominance for binary + WordType
        run_dominance_by_group()

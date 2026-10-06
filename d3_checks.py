# -*- coding: utf-8 -*-
"""d3_checks.py

Read-only checks (no model fitting):
  1. Baseline binary metrics (outputs/baseline/binary_metrics.csv) vs. report values.
  2. Median imputation value of each PN column vs. per-WordType observed medians.
  3. Precision / coverage of the rule "PN missing -> oral".

Data loading reuses diagnostics_pn.py, which extracts load_data() and constants from
prediction.py's source text (prediction.py is never imported). Importing diagnostics_pn
only parses prediction.py; it writes nothing. Outputs: outputs/diagnostics/d3_*.
"""

import os
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

DESKTOP = os.path.expanduser('~/Desktop')
sys.path.insert(0, DESKTOP)
import diagnostics_pn as D  # noqa: E402

OUT_DIR = os.path.join(DESKTOP, 'outputs', 'diagnostics')
BASELINE_DIR = os.path.join(DESKTOP, 'outputs', 'baseline')
PN_COLS = D.PN_FEATURE_COLS
WORDTYPES = ['CV', 'PNV', 'V']
GROUPS = ['CI', 'NH', 'All']

REPORT = {
    'full': {'Accuracy': np.nan, 'AUC': np.nan},
    'CI': {'Accuracy': 0.879, 'AUC': 0.954},
    'NH': {'Accuracy': 0.944, 'AUC': 0.981},
}


def save(df, name):
    path = os.path.join(OUT_DIR, name)
    assert os.path.realpath(path).startswith(os.path.realpath(OUT_DIR) + os.sep)
    df.to_csv(path, index=False)
    print(f'[saved] {path}')


def item1():
    D.header('1. BASELINE BINARY METRICS vs REPORT')
    path = os.path.join(BASELINE_DIR, 'binary_metrics.csv')
    bm = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['Run'])
    print(f'Read: {path} (mtime {pd.Timestamp(os.path.getmtime(path), unit="s")} UTC)'
          if os.path.exists(path) else f'Missing: {path}')
    rows = []
    for run in ['full', 'CI', 'NH']:
        r = bm[bm['Run'] == run]
        rep = REPORT[run]
        if r.empty:
            rows.append({'Run': run, 'Status': 'not available', 'N': np.nan,
                         'Accuracy': np.nan, 'Report accuracy': rep['Accuracy'],
                         'AUC': np.nan, 'Report AUC': rep['AUC'],
                         'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan,
                         'Accuracy match (3dp)': 'n/a', 'AUC match (3dp)': 'n/a'})
            continue
        r = r.iloc[0]

        def verdict(metric):
            if np.isnan(rep[metric]):
                return 'no report value'
            return 'yes' if round(float(r[metric]), 3) == rep[metric] else 'NO'

        rows.append({'Run': run, 'Status': 'available', 'N': int(r['N']),
                     'Accuracy': round(float(r['Accuracy']), 3),
                     'Report accuracy': rep['Accuracy'],
                     'AUC': round(float(r['AUC']), 3), 'Report AUC': rep['AUC'],
                     'F1': round(float(r['F1']), 3),
                     'Precision': round(float(r['Precision']), 3),
                     'Recall': round(float(r['Recall']), 3),
                     'Accuracy match (3dp)': verdict('Accuracy'),
                     'AUC match (3dp)': verdict('AUC')})
    t = pd.DataFrame(rows)
    D.show(t.fillna(''), index=False)
    print("Note: the report and the baseline both use N=1421 (CI 793 / NH 628; 'Exclude? Reason.'")
    print('rows dropped), so mismatches stem from the pipeline, not the sample.')
    save(t, 'd3_binary_metrics_vs_report.csv')


def item2(df):
    D.header('2. PN MEDIAN-IMPUTATION VALUE vs PER-WORDTYPE OBSERVED MEDIANS')
    print('Overall median = median of all observed values in the group (approximately what')
    print("SimpleImputer(strategy='median') imputes; the real pipeline fits it on training folds only).")
    rows = []
    for c in PN_COLS:
        for g in GROUPS:
            sub = df.loc[D.group_mask(df, g)]
            obs = sub[c].dropna().astype(float)
            row = {'PN column': c, 'Group': g, 'N obs': len(obs),
                   'Overall median': np.median(obs) if len(obs) else np.nan}
            for w in WORDTYPES:
                v = sub.loc[sub['WordType'] == w, c].dropna().astype(float)
                row[f'N obs {w}'] = len(v)
                row[f'Median {w}'] = np.median(v) if len(v) else np.nan
            dist = {w: abs(row['Overall median'] - row[f'Median {w}'])
                    for w in WORDTYPES if not np.isnan(row[f'Median {w}'])}
            row['Closest WordType'] = min(dist, key=dist.get) if dist else ''
            row['|diff| to closest'] = min(dist.values()) if dist else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    cols = (['PN column', 'Group', 'N obs', 'Overall median']
            + [f'Median {w}' for w in WORDTYPES] + ['Closest WordType', '|diff| to closest']
            + [f'N obs {w}' for w in WORDTYPES])
    t = t[cols]
    D.show(t.round(2), index=False)
    save(t, 'd3_pn_impute_median_vs_wordtype.csv')


def item3(df):
    D.header('3. "PN MISSING -> ORAL": PRECISION AND COVERAGE')
    n_miss = df[PN_COLS].isna().sum(axis=1)
    partial = int(((n_miss > 0) & (n_miss < len(PN_COLS))).sum())
    print(f'PN_missing = 1 if all {len(PN_COLS)} PN columns are missing '
          f'(partially-missing rows: {partial}). Oral = PNV + V (label 0).')
    df = df.assign(PN_missing=(n_miss == len(PN_COLS)).astype(int))
    rows = []
    for g in GROUPS:
        sub = df.loc[D.group_mask(df, g)]
        miss = sub['PN_missing'] == 1
        oral = sub['label'] == 0
        row = {'Group': g, 'N': len(sub),
               'n PN_missing=1': int(miss.sum()),
               'n PN_missing=1 & oral': int((miss & oral).sum()),
               '% PN_missing that are oral (precision)':
                   100 * (miss & oral).sum() / miss.sum() if miss.any() else np.nan,
               'n oral': int(oral.sum()),
               '% oral with PN_missing (coverage)':
                   100 * (miss & oral).sum() / oral.sum() if oral.any() else np.nan}
        for w in ['PNV', 'V']:
            m = sub['WordType'] == w
            row[f'n {w}'] = int(m.sum())
            row[f'n {w} PN_missing'] = int((m & miss).sum())
            row[f'% {w} with PN_missing'] = 100 * (m & miss).sum() / m.sum() if m.any() else np.nan
        row['n CV PN_missing'] = int((miss & (sub['WordType'] == 'CV')).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    D.show(t.round(1), index=False)
    save(t, 'd3_pn_missing_precision_coverage.csv')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = D.P['_default_sheet_url']()
    print(f'Source: {src}')
    print(f'Loader: load_data() extracted from {D.PREDICTION_PY} via diagnostics_pn.py')
    print(f'PN columns: {PN_COLS}')
    df = D.load_data(D.read_csv_maybe_ssl(src))
    counts = df['Group_1'].value_counts()
    print(f'N = {len(df)} (CI {counts.get("CI", 0)}, NH {counts.get("NH", 0)})')
    assert (len(df), counts.get('CI'), counts.get('NH')) == (1421, 793, 628), \
        'Data does not match the baseline load (expected 1421 / 793 / 628)'
    item1()
    item2(df)
    item3(df)


if __name__ == '__main__':
    main()

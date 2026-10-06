# -*- coding: utf-8 -*-
"""diagnostics_pn.py

Read-only diagnostics for the PN_* (prenasalization) features used by prediction.py.
No models are fit. All outputs go to outputs/diagnostics/ next to this file.

prediction.py is NOT imported: its module top level calls os.makedirs(FIG_DIR) and sets
plotting state. Instead, the data-loading functions and constants are pulled out of its
source with `ast` and executed in an isolated namespace, so this script uses the exact
same load_data() as the baseline without running any of its top-level code.
"""

import ast
import os
import ssl
import sys
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTION_PY = os.path.join(HERE, 'prediction.py')
OUT_DIR = os.path.join(HERE, 'outputs', 'diagnostics')
BASELINE_DIR = os.path.join(HERE, 'outputs', 'baseline')

_FUNCS = {'_pct', '_resolve_col', '_default_sheet_url', 'read_csv_maybe_ssl', 'load_data'}
_CONSTS = {'PN_FEATURE_COLS', 'CLEAN_FEATURE_COLS', 'NO_PN_FEATURE_COLS',
           'SPEAKER_COL_CANDIDATES', 'GROUP_COL_CANDIDATES', 'SUBSTITUTION_COL',
           'WORDTYPE_ORDER'}


def _extract_from_prediction(path=PREDICTION_PY):
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=path)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in _CONSTS for t in node.targets):
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ns = {'np': np, 'pd': pd, 'os': os, 'ssl': ssl, 'urllib': urllib,
          'StringIO': StringIO, '__name__': 'prediction_extract'}
    exec(compile(module, path, 'exec'), ns)
    missing = (_FUNCS | _CONSTS) - set(ns)
    if missing:
        raise RuntimeError(f'Could not extract from prediction.py: {sorted(missing)}')
    return ns


P = _extract_from_prediction()
load_data = P['load_data']
read_csv_maybe_ssl = P['read_csv_maybe_ssl']
PN_FEATURE_COLS = P['PN_FEATURE_COLS']
SUBSTITUTION_COL = P['SUBSTITUTION_COL']
WORDTYPE_ORDER = P['WORDTYPE_ORDER']
EXCLUDE_COL = 'Exclude? Reason.'
GROUPS = ['CI', 'NH', 'All']

REPORT_COUNTS = pd.DataFrame(
    {'CV': [311, 236], 'PNV': [242, 197], 'V': [240, 195]}, index=['CI', 'NH'])


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def header(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def show(df, title=None, index=True):
    if title:
        print(f'\n{title}')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           'display.width', 250, 'display.max_colwidth', 70):
        print(df.to_string(index=index))


def save(df, name, index=False):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=index)
    print(f'[saved] {path}')


def group_mask(df, g):
    return np.ones(len(df), dtype=bool) if g == 'All' else (df['Group_1'] == g).values


def wt_crosstab(df):
    ct = pd.crosstab(df['Group_1'], df['WordType'])
    ct = ct.reindex(columns=[w for w in WORDTYPE_ORDER if w in ct.columns]
                    + sorted(set(ct.columns) - set(WORDTYPE_ORDER)), fill_value=0)
    ct['Total'] = ct.sum(axis=1)
    ct.loc['All'] = ct.sum()
    return ct


# ---------------------------------------------------------------------------
# 1. Row-count reconciliation
# ---------------------------------------------------------------------------
def section1(raw, df):
    header('1. RECONCILE ROW COUNTS')

    raw_g = raw['Group_1'].fillna('<missing>')
    no_wt = raw['WordType'].isna()
    a = pd.DataFrame({
        'Raw rows': raw_g.value_counts(),
        'Set aside (missing WordType)': raw_g[no_wt].value_counts(),
        'Final N (after load_data)': df['Group_1'].fillna('<missing>').value_counts(),
    }).fillna(0).astype(int)
    a.loc['All'] = a.sum()
    show(a, '(a) Raw rows, rows set aside for missing WordType, final N, by Group_1:')
    save(a, 'rowcounts_a_raw_vs_final.csv', index=True)

    b = wt_crosstab(df)
    show(b, '(b) N by Group_1 x WordType after load_data() (Exclude-noted rows KEPT):')
    save(b, 'rowcounts_b_group_x_wordtype_loaddata.csv', index=True)

    excl = df[EXCLUDE_COL].notna()
    c = wt_crosstab(df[excl])
    show(c, f"(c) Rows with a non-null '{EXCLUDE_COL}' note, by Group_1 x WordType:")
    save(c, 'rowcounts_c_exclude_noted.csv', index=True)

    d = wt_crosstab(df[~excl])
    show(d, f"(d) N by Group_1 x WordType with '{EXCLUDE_COL}' rows REMOVED:")
    save(d, 'rowcounts_d_group_x_wordtype_exclude_removed.csv', index=True)

    rows = []
    for name, tab in [('load_data (Exclude rows kept)', b), ('Exclude rows removed', d)]:
        cells = all(int(tab.loc[g, w]) == int(REPORT_COUNTS.loc[g, w])
                    for g in REPORT_COUNTS.index for w in REPORT_COUNTS.columns)
        rows.append({
            'Version': name,
            'N': int(tab.loc['All', 'Total']),
            'N CI': int(tab.loc['CI', 'Total']), 'N NH': int(tab.loc['NH', 'Total']),
            'V CI': int(tab.loc['CI', 'V']), 'V NH': int(tab.loc['NH', 'V']),
            'N=1421 (793/628)?': (int(tab.loc['All', 'Total']), int(tab.loc['CI', 'Total']),
                                  int(tab.loc['NH', 'Total'])) == (1421, 793, 628),
            'V 240/195?': (int(tab.loc['CI', 'V']), int(tab.loc['NH', 'V'])) == (240, 195),
            'V 254/198?': (int(tab.loc['CI', 'V']), int(tab.loc['NH', 'V'])) == (254, 198),
            'All 6 report cells?': cells,
        })
    match = pd.DataFrame(rows)
    show(match, '(d) Which version matches the report?', index=False)
    save(match, 'rowcounts_d_version_match.csv')

    print('\n(e) Version used by the baseline run (read-only evidence):')
    print("  main() -> load_data(raw_df): keeps 'Exclude? Reason.' rows, drops only rows with")
    print('  missing WordType; main() then only subsets by Group_1.')
    log_path = os.path.join(BASELINE_DIR, 'baseline_log.txt')
    if os.path.exists(log_path):
        with open(log_path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        keys = ('=== Baseline run', 'Loaded ', 'Keeping ', 'Setting aside', 'N = ')
        for i, line in enumerate(lines, start=1):
            if line.startswith(keys):
                print(f'    baseline_log.txt:{i}: {line.rstrip()}')
    for name in ('binary_metrics.csv', 'wordtype_accuracy.csv', 'n_support.csv'):
        path = os.path.join(BASELINE_DIR, name)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                print(f'    {name}:')
                for line in f.read().strip().splitlines():
                    print(f'      {line}')


# ---------------------------------------------------------------------------
# 2. PN missingness by WordType
# ---------------------------------------------------------------------------
def section2(df):
    header('2. PN MISSINGNESS BY WORDTYPE')
    n_miss = df[PN_FEATURE_COLS].isna().sum(axis=1)
    k = len(PN_FEATURE_COLS)
    rows = []
    for g in GROUPS:
        gm = group_mask(df, g)
        for w in WORDTYPE_ORDER + ['All']:
            m = gm & ((df['WordType'] == w).values if w != 'All' else True)
            n = int(m.sum())
            all_present = int((n_miss[m] == 0).sum())
            all_missing = int((n_miss[m] == k).sum())
            partial = int(((n_miss[m] > 0) & (n_miss[m] < k)).sum())
            rows.append({
                'Group': g, 'WordType': w, 'N': n,
                'All4 present': all_present, 'All4 missing': all_missing,
                'Partial': partial,
                '% all missing': round(100 * all_missing / n, 1) if n else np.nan,
                '% any missing': round(100 * (all_missing + partial) / n, 1) if n else np.nan,
            })
    t = pd.DataFrame(rows)
    show(t, index=False)
    save(t, 'pn_missingness_by_wordtype.csv')

    for w in ('CV', 'PNV'):
        mw = df['WordType'] == w
        mm = mw & (n_miss > 0)
        print(f'{w} tokens with >=1 missing PN value: {int(mm.sum())}/{int(mw.sum())} '
              f'(all 4 missing: {int((mw & (n_miss == k)).sum())})')
        if mm.any():
            cols = ['Speaker', 'Group_1', 'WordType', 'RAND'] + PN_FEATURE_COLS
            show(df.loc[mm, cols], f'  {w} tokens with missing PN:', index=False)
    return t


# ---------------------------------------------------------------------------
# 3. Scale / range of observed PN
# ---------------------------------------------------------------------------
def section3(df):
    header('3. SCALE / RANGE OF OBSERVED PN VALUES')
    overall = pd.DataFrame([{
        'PN column': c, 'N observed': int(df[c].notna().sum()),
        'Min': df[c].min(), 'Max': df[c].max(),
    } for c in PN_FEATURE_COLS])
    show(overall, 'Overall observed min / max:', index=False)
    save(overall, 'pn_overall_min_max.csv')

    rows = []
    for c in PN_FEATURE_COLS:
        for g in GROUPS:
            gm = group_mask(df, g)
            for w in WORDTYPE_ORDER:
                v = df.loc[gm & (df['WordType'] == w).values, c].dropna().astype(float).values
                if len(v):
                    st = {'Min': v.min(), 'P5': np.percentile(v, 5), 'Median': np.median(v),
                          'P95': np.percentile(v, 95), 'Max': v.max(),
                          'N<=0': int((v <= 0).sum())}
                else:
                    st = {'Min': np.nan, 'P5': np.nan, 'Median': np.nan, 'P95': np.nan,
                          'Max': np.nan, 'N<=0': 0}
                rows.append({'PN column': c, 'Group': g, 'WordType': w, 'N obs': len(v), **st})
    t = pd.DataFrame(rows)
    show(t.round(3), 'Observed PN by Group_1 x WordType:', index=False)
    save(t, 'pn_observed_range_by_wordtype.csv')
    return t


# ---------------------------------------------------------------------------
# 4. Observed PN on non-PNV tokens
# ---------------------------------------------------------------------------
def _is_initial_note(note):
    note = note.fillna('').astype(str).str.lower()
    return note.str.contains('init') | note.str.contains(r'^int ')


def section4(df):
    header('4. OBSERVED PN ON NON-PNV (CV and V) TOKENS')
    has_obs = df[PN_FEATURE_COLS].notna().any(axis=1)
    sel = has_obs & df['WordType'].isin(['CV', 'V'])
    out = df.loc[sel].copy()
    out['SubstitutionNote_raw'] = out[SUBSTITUTION_COL] if SUBSTITUTION_COL in out else np.nan
    cols = ['Speaker', 'Group_1', 'WordType', 'RAND', 'PRODUCTION', 'ActualProduction',
            'ActualManner', 'InitialSubstitution', 'SubstitutionNote_raw'] + PN_FEATURE_COLS
    if EXCLUDE_COL in out:
        out['ExcludeNote'] = out[EXCLUDE_COL]
        cols.append('ExcludeNote')
    out = out[[c for c in cols if c in out.columns]].sort_values(
        ['Group_1', 'Speaker', 'WordType']).reset_index(drop=True)
    print("'InitialSubstitution' = load_data()'s derived column (note text only for initial")
    print("'nasal to oral' / 'oral to nasal' / 'oral to prenas' substitutions, else '').")
    print(f"'SubstitutionNote_raw' = raw '{SUBSTITUTION_COL}' column.")
    print(f'N = {len(out)} CV/V tokens with >=1 observed PN value '
          f'(CV: {int((out.WordType == "CV").sum())}, V: {int((out.WordType == "V").sum())}); '
          f'all 4 observed: {int(df.loc[sel, PN_FEATURE_COLS].notna().all(axis=1).sum())}')
    show(out, index=False)
    save(out, 'pn_observed_in_nonPNV.csv')

    by_spk = (pd.crosstab([out['Group_1'], out['Speaker']], out['WordType'])
              .reindex(columns=['CV', 'V'], fill_value=0))
    by_spk['Total'] = by_spk.sum(axis=1)
    by_spk.loc[('All', ''), :] = by_spk.sum()
    by_spk = by_spk.astype(int)
    show(by_spk, 'Count by Group_1 x Speaker x WordType:')
    save(by_spk, 'pn_observed_in_nonPNV_counts_by_speaker.csv', index=True)

    by_grp = (pd.crosstab(out['Group_1'], out['WordType'])
              .reindex(index=['CI', 'NH'], columns=['CV', 'V'], fill_value=0))
    by_grp['Total'] = by_grp.sum(axis=1)
    by_grp.loc['All'] = by_grp.sum()
    show(by_grp, 'Count by Group_1 x WordType:')
    save(by_grp, 'pn_observed_in_nonPNV_counts_by_group.csv', index=True)

    raw_note = out['SubstitutionNote_raw'].fillna('').astype(str).str.strip()
    init_note = _is_initial_note(out['SubstitutionNote_raw'])
    derived = out['InitialSubstitution'].fillna('').astype(str) != ''
    rows = []
    for w in ['CV', 'V', 'All']:
        m = (out['WordType'] == w) if w != 'All' else pd.Series(True, index=out.index)
        rows.append({
            'WordType': w, 'N': int(m.sum()),
            'Any raw note': int((m & (raw_note != '')).sum()),
            'No raw note': int((m & (raw_note == '')).sum()),
            'Initial-position note': int((m & init_note).sum()),
            'load_data InitialSubstitution set': int((m & derived).sum()),
            'ActualManner=nasal': int((m & (out['ActualManner'] == 'nasal')).sum()),
            'ActualManner=oral': int((m & (out['ActualManner'] == 'oral')).sum()),
        })
    notes = pd.DataFrame(rows)
    show(notes, 'Substitution-note counts among CV/V tokens with observed PN:', index=False)
    save(notes, 'pn_observed_in_nonPNV_substitution_counts.csv')


# ---------------------------------------------------------------------------
# 5. Power of the PN-missing flag (no model is fit)
# ---------------------------------------------------------------------------
def section5(df):
    header('5. PN_missing FLAG POWER (no model is fit)')
    n_miss = df[PN_FEATURE_COLS].isna().sum(axis=1)
    k = len(PN_FEATURE_COLS)
    pn_missing = (n_miss == k).astype(int)
    n_partial = int(((n_miss > 0) & (n_miss < k)).sum())
    print(f'PN_missing = 1 if all {k} PN columns are missing; partially-missing rows: {n_partial} '
          f'(PN_missing=0).')

    crosstabs, rows = [], []
    for g in GROUPS:
        gm = group_mask(df, g)
        sub = df.loc[gm]
        flag = pn_missing[gm].values
        ct = (pd.crosstab(pd.Series(flag, name='PN_missing'), sub['WordType'].values)
              .reindex(index=[0, 1], columns=WORDTYPE_ORDER, fill_value=0))
        ct['Total'] = ct.sum(axis=1)
        show(ct, f'[{g}] PN_missing x WordType:')
        crosstabs.append(ct.reset_index().assign(Group=g))

        y = sub['label'].values
        auc = roc_auc_score(y, flag)
        rule_a = np.where(flag == 1, 0, 1)
        rule_b = 1 - rule_a
        rows.append({
            'Group': g, 'N': len(y), 'N nasal(CV)': int(y.sum()),
            'Majority acc': round(max(y.mean(), 1 - y.mean()), 4),
            'AUC(label,PN_missing)': round(auc, 4),
            'Flipped AUC (1-AUC)': round(1 - auc, 4),
            'Acc rule A (missing->oral, else nasal)': round(float((rule_a == y).mean()), 4),
            'Acc rule B (missing->nasal, else oral)': round(float((rule_b == y).mean()), 4),
        })
    summary = pd.DataFrame(rows)
    show(summary, 'Flag power by group (label: CV=1, PNV/V=0):', index=False)
    print('AUC(label, PN_missing) uses PN_missing as a score for label=1 (CV). '
          'Flipped AUC = AUC of 1-PN_missing.')

    wt = df['WordType']
    share = {w: float(pn_missing[(wt == w).values].mean()) for w in WORDTYPE_ORDER}
    print('Share of tokens with PN_missing=1 by WordType (All): '
          + ', '.join(f'{w} {100 * s:.1f}%' for w, s in share.items()))
    save(pd.concat(crosstabs, ignore_index=True)[['Group', 'PN_missing'] + WORDTYPE_ORDER + ['Total']],
         'pn_missing_flag_crosstab.csv')
    save(summary, 'pn_missing_flag_power.csv')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log = open(os.path.join(OUT_DIR, 'diagnostics_log.txt'), 'w', encoding='utf-8')
    orig = sys.stdout
    sys.stdout = Tee(orig, log)
    try:
        src = P['_default_sheet_url']()
        print(f'Source: {src}')
        print(f'Functions/constants extracted (not imported) from: {PREDICTION_PY}')
        raw = read_csv_maybe_ssl(src)
        df = load_data(raw)
        section1(raw, df)
        section2(df)
        section3(df)
        section4(df)
        section5(df)
        print(f'\nAll diagnostics written to {OUT_DIR}/')
    finally:
        sys.stdout = orig
        log.close()


if __name__ == '__main__':
    main()

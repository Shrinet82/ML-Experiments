import json

with open('tabpfn_benchmark.py', 'r') as f:
    py_code = f.read()

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '# TabPFN: Multi-Dataset Statistical Benchmark & Business Translation\n',
            '**Paper Reference:** Hollmann et al., *Nature* 637, 319-326 (Jan 2025)  \n',
            '**Statistical Perspective:** Zhang, Tan, Tian, Li (*arXiv:2505.20003*, May 2025)  \n',
            '**Core Mechanism:** Müller et al., *Transformers Can Do Bayesian Inference* (ICLR 2022)  \n',
            '\n',
            '---\n',
            '### Evaluation Protocol Highlights:\n',
            '- **Model Weights:** TabPFN v2 via Prior Labs License (Apache-2.0 + Attribution)\n',
            '- **Datasets:** `credit-g` (31), `telco-churn` (42178), `bank-marketing` (1461, N=10k), `adult` (1590, N=10k)\n',
            '- **Statistical Rigor:** Repeated Stratified CV (5 Folds x 5 Repeats = 25 evaluations per model), exact identical splits, 95% CIs, Wilcoxon signed-rank tests, Nadeau & Bengio corrected resampled t-tests\n',
            '- **Fair Baselines:** CatBoost native `cat_features`, LightGBM categorical dtypes, XGBoost ordinal encoded, nested 15-trial tuning budget on internal validation splits (zero test leakage)\n',
            '- **Business Translation:** German Credit cost matrix (FP=5, FN=1) & Telco Churn profit optimization ($100 retained vs $20 cost)\n',
            '- **Selective Classification & Calibration:** Predictive entropy rejection curves, Brier score, ECE, and reliability diagrams\n',
            '- **Scaling Analysis:** AUC vs sample size N_train in [200, 10000]\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            '# Ensure dependencies are installed in execution environment\n',
            '!pip install --quiet "tabpfn>=2.0.0" catboost xgboost lightgbm scikit-learn seaborn matplotlib\n'
        ]
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            py_code
        ]
    }
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('tabpfn_benchmark.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Successfully created tabpfn_benchmark.ipynb cleanly!')

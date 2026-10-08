import json

with open("chronos2_benchmark.py", "r") as f:
    py_code = f.read()

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Chronos-2 vs Standard Baselines on M5 Retail Demand\n",
            "### Empirical Foundation Model Replication & Statistical Evaluation\n",
            "\n",
            "> **Paper Reference:** *Chronos-2: From Univariate to Universal Forecasting*  \n",
            "> **Authors:** Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, Michael Bohlke-Schneider (*Amazon Web Services AI Labs*, arXiv:2510.15821, Oct 2025)  \n",
            "> **License:** Apache-2.0 | **Model ID:** `amazon/chronos-2` (120M parameters)  \n",
            "> **Evaluation Protocol:** 1,000 item-store series from Walmart M5 retail demand, 3 rolling test origins (28-day forecast horizon each), paired bootstrap significance testing (10,000 resamples), asymmetric Newsvendor inventory cost evaluation.\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 0: Install dependencies in Kaggle environment\n",
            "!pip install --quiet \"chronos-forecasting>=2.0\" statsforecast lightgbm \"pandas[pyarrow]\"\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            py_code
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Execute Benchmark Pipeline with Step 1 Smoke Test & Step 2 Full Execution\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "data_dir, _, _, _ = locate_m5_data()\n",
            "output_dir = Path('/kaggle/working') if os.path.exists('/kaggle/working') else Path('results')\n",
            "\n",
            "print('>>> STEP 1: SMOKE TEST ON 20 SERIES (1 ORIGIN) <<<')\n",
            "smoke_dir = output_dir / 'smoke_test'\n",
            "run_benchmark(data_dir=data_dir, output_dir=smoke_dir, smoke_test=True)\n",
            "assert (smoke_dir / 'per_series_metrics.csv').exists(), 'Smoke test failed: per_series_metrics.csv missing'\n",
            "assert (smoke_dir / 'summary.csv').exists(), 'Smoke test failed: summary.csv missing'\n",
            "print('>>> STEP 1 SMOKE TEST PASSED! PROCEEDING TO FULL 1,000-SERIES RUN <<<')\n",
            "\n",
            "print('>>> STEP 2: FULL RUN (1,000 SERIES, 3 ROLLING ORIGINS) <<<')\n",
            "run_benchmark(data_dir=data_dir, output_dir=output_dir, smoke_test=False)\n",
            "print('>>> STEP 2 FULL BENCHMARK RUN COMPLETED! ALL ARTIFACTS WRITTEN. <<<')\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("chronos2_benchmark.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Successfully generated chronos2_benchmark.ipynb!")

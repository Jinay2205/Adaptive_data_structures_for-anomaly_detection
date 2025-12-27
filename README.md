# Adaptive Probabilistic Data Structures for Anomaly Detection in Non-Stationary Data Streams

This repository contains the code and experimental framework for the
project:

**"A Comparative Analysis of Adaptive Probabilistic Data Structures for
Anomaly Detection in Non-Stationary Data Streams"**

This project evaluates the efficacy of *hybrid* and *adaptive*
Probabilistic Data Structures (PDS) against static baselines using
real-world streaming data from the Numenta Anomaly Benchmark (NAB).

------------------------------------------------------------------------

## Contents

-   **models.py** -- Implementations of:
    -   Count-Min Sketch (CMS)
    -   Ada-Sketch
    -   Sliding Window CBF
    -   Stable Learned Bloom Filter (SLBF)
-   **main.py** -- Runs the full experiment and generates:
    -   `final_project_plot.png`
    -   `final_metrics.json`
-   **utils.py** -- Data loading, oracle ground truth generation,
    evaluation metrics\
-   **art_daily_jumpsup.csv** -- NAB dataset\
-   **Makefile** -- Automates environment setup and execution\
-   **requirements.txt** -- Dependencies list

------------------------------------------------------------------------

## Prerequisites

-   Python 3.x\
-   numpy\
-   pandas\
-   matplotlib\
-   scikit-learn

------------------------------------------------------------------------

## How to Run

### Option 1: Quick Start (Using Makefile)

``` bash
make install
make run
```

### Option 2: Manual Execution (Windows / No Makefile)

**Step 1: Create a virtual environment**

``` bash
python -m venv env
```

**Step 2: Activate the environment**

macOS / Linux:

``` bash
source env/bin/activate
```

Windows:

``` bash
env\Scripts\activate
```

**Step 3: Install dependencies**

``` bash
pip install -r requirements.txt
```

**Step 4: Run the experiment**

``` bash
python main.py
```

------------------------------------------------------------------------

## Outputs

### `final_project_plot.png`

A 3-panel figure showing: - Oracle anomalies\
- CMS vs Ada-Sketch scores\
- SLBF detections

### `final_metrics.json`

Precision, Recall, F1 for all four streaming models.

------------------------------------------------------------------------



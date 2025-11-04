# A Benchmarking Framework for Extreme Class Imbalance in Smart Grid Predictive Maintenance

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the official implementation for the paper: **"A Benchmarking Framework for Extreme Class Imbalance in Smart Grid Predictive Maintenance: Preliminary Evidence from Rare Failure Event Detection"** by Amirkiavan Shafiei and Shima Sharifi.

This repository provides the complete code for synthetic data generation, the rigorous temporal validation framework, and the evaluation of machine learning models under realistic "cold-start" scenarios for predicting rare equipment failures in smart grids.

## Abstract

> This work establishes a rigorous, standards-compliant benchmark framework for evaluating predictive maintenance models under realistic extreme class imbalance conditions (failure rates < 0.1%) in smart grid applications, employing strict temporal validation to simulate ``cold-start'' operational scenarios where no failure events are present in training data. Using a transformer dataset with 18 failure events among 60,000 samples (0.030\% failure rate), our framework addresses critical methodological gaps in prior literature: inconsistent evaluation protocols that mask data leakage, inadequate analysis of operational trade-offs, and weak interoperability standards. Our evaluation reveals a profound performance paradox: all supervised learning methods---including cost-sensitive SVM, resampling techniques (e.g., Temporal-SMOTE), and deep learning (LSTM)---failed completely, achieving zero recall and F1-score on unseen failures. In contrast, an unsupervised One-Class SVM achieved perfect technical success with 1.0 recall by learning normal operational boundaries. However, this came at the cost of near-zero precision ($\approx$0.003) and a catastrophic 95.5\% false alarm rate, leading to economically unviable outcomes. The primary contribution is the reproducible framework itself, which provides the first honest performance baseline for this domain. These findings demonstrate that current state-of-the-art models are not ready for operational deployment in cold-start scenarios and motivate a shift toward hybrid architectures that combine high-sensitivity anomaly detection with secondary verification mechanisms to manage false alarm costs.

## Key Findings

Our framework's core achievement was uncovering a **Performance-Economic Paradox**:

1.  **Complete Failure of Supervised Learning:** In our realistic "cold-start" test (0 failures in the training set), all supervised models (RF, LSTM, Cost-Sensitive SVM) achieved **0.0 Recall**. They are fundamentally unable to predict failure modes they have never seen.
2.  **Technical Success of Anomaly Detection:** An unsupervised `One-Class SVM` was the only model to successfully detect failures, achieving a **perfect 1.0 Recall**.
3.  **Economic Catastrophe:** This technical success was coupled with a **~95.5% false alarm rate**, leading to a catastrophic negative ROI of over -600,000% and demonstrating that the model is operationally unusable as a standalone solution.

## Framework Architecture

The code is organized in a modular structure to ensure clarity and reproducibility:

-   `main.py`: The main execution script that orchestrates the entire benchmark.
-   `config.py`: Central configuration file for all dataset, model, and cost parameters.
-   `temporal_validation.py`: Implements the strict, "cold-start" temporal data split to prevent data leakage.
-   `data_processor.py` & `utils.py`: Contain logic for generating the synthetic dataset with realistic physical and temporal patterns.
-   `imbalance_handlers.py`: Contains implementations of the evaluated models, including the temporally-enhanced `OneClassTemporalSVM`.
-   `evaluation_metrics.py`: Calculates all performance and economic impact metrics.
-   `visualization.py`: Generates all figures used in the paper.

## Installation

To set up the environment and run the code, please follow these steps. This project was developed using Python 3.10.

**1. Clone the repository:**
```bash
git clone https://github.com/keivanShafiei/Predictive-Maintenance-in-Smart-Grids.git
cd Predictive-Maintenance-in-Smart-Grids
```

**2. Create and activate a virtual environment (recommended):**
*   On macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
*   On Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

**3. Install the required packages:**
This repository includes a `requirements.txt` file that lists all necessary dependencies.
```bash
pip install -r requirements.txt
```

## Usage

To reproduce the main results of the paper, simply run the main script. The framework is configured to execute the complete "cold-start" benchmark by default.

```bash
python src/main.py
```

### Expected Output

Running the script will:
1.  Generate the synthetic dataset according to the parameters in `config.py`.
2.  Perform the temporal split, model training, and evaluation.
3.  Print live progress and key results to the console.
4.  Generate a comprehensive `smart_grid_benchmark_report.txt` file in the root directory.
5.  Create a `figures/` directory containing all the plots from the paper in PNG format:
    -   `performance_comparison.png`
    -   `economic_analysis.png`
    -   `temporal_validation_comparison.png`
    -   `cost_sensitivity_analysis.png`
    -   `precision_recall_curves.png`


## License

This project is licensed under the **Apache 2.0 License**. Please see the `LICENSE` file for full details.

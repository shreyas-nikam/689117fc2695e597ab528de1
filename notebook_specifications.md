
# Jupyter Notebook Specification: NPV Model Validation Simulator

## 1. Notebook Overview

**Learning Goals:**

*   Understand the end-to-end process of NPV model validation for loan valuation.
*   Learn how to perform analytical test cases, scenario testing, and back-testing.
*   Explore the impact of different risk factors and governance triggers on model outcomes.
*   Learn how to generate an automated model validation report.

**Expected Outcomes:**

*   A functional simulator that can perform formula verification, scenario testing, back-testing, and statistical analysis on an NPV model.
*   Visualizations that illustrate the results of these tests and analyses.
*   A governance dashboard that tracks threshold breaches and open issues.
*   A generated model validation report documenting methodology, findings, required actions, and sign-off workflow.

## 2. Mathematical and Theoretical Foundations

### 2.1 Net Present Value (NPV)

The Net Present Value (NPV) is the sum of the present values of incoming and outgoing cash flows over a period of time.  It is calculated using the following formula:

$$NPV = \sum_{t=0}^{N} \frac{CF_t}{(1+r)^t}$$

Where:
*   $CF_t$ is the cash flow at time *t*.
*   *r* is the discount rate (cost of capital).
*   *t* is the time period.
*   *N* is the total number of time periods.

**Real-world application:** NPV is used to determine the profitability of an investment or project.  A positive NPV indicates that the investment is expected to be profitable, while a negative NPV indicates that it is expected to result in a loss.

**Derivation (simplified):** The formula discounts future cash flows back to their present value using the discount rate, which reflects the time value of money and the risk associated with the cash flows. The summation calculates the present value of all cash flows, yielding the NPV.

### 2.2 Delta NPV (ΔNPV)
Delta NPV represents the change in Net Present Value between two scenarios (baseline and challenged).  It is calculated by subtracting the baseline NPV from the challenged NPV:

$$ \Delta NPV = NPV_{Challenged} - NPV_{Baseline} $$

A **zero change scenario** should yield a $\Delta NPV$ of zero.

**Real-world application:** Assessing the impact of changes in input parameters (e.g., interest rates, prepayment speeds) on the loan valuation.

### 2.3 Root Mean Squared Error (RMSE)

The Root Mean Squared Error (RMSE) is a measure of the differences between values predicted by a model and the values observed. It's the square root of the average of the squared differences.

$$RMSE = \sqrt{\frac{\sum_{i=1}^{n}(P_i - O_i)^2}{n}}$$

Where:
*   $P_i$ is the predicted value for the i-th observation.
*   $O_i$ is the observed value for the i-th observation.
*   *n* is the number of observations.

**Real-world application:** Used in back-testing to assess the accuracy of NPV model predictions against realized cash flows.

### 2.4 Mean Absolute Error (MAE)

The Mean Absolute Error (MAE) is a measure of the differences between values predicted by a model and the values observed. It's the average of the absolute differences.

$$MAE = \frac{\sum_{i=1}^{n}|P_i - O_i|}{n}$$

Where:
*   $P_i$ is the predicted value for the i-th observation.
*   $O_i$ is the observed value for the i-th observation.
*   *n* is the number of observations.

**Real-world application:** Used as an alternative to RMSE in back-testing. Less sensitive to outliers than RMSE.

### 2.5 Diebold-Mariano Test

The Diebold-Mariano (DM) test is a statistical test used to compare the forecast accuracy of two competing forecasting models. The null hypothesis is that the two models have equal forecast accuracy.  A significant DM statistic indicates that one model has significantly better forecast accuracy than the other.

The DM statistic is calculated as:

$$ DM = \frac{\bar{d}}{\sqrt{\frac{2\pi \hat{f}_d(0)}{T}}} $$

Where:

*   $\bar{d}$ is the mean of the loss differential series $d_t = L(e_{1t}) - L(e_{2t})$, where $e_{1t}$ and $e_{2t}$ are the forecast errors of the two models at time t, and $L(.)$ is a loss function (e.g., squared error or absolute error).
*   $\hat{f}_d(0)$ is a consistent estimate of the spectral density of $d_t$ at frequency zero.
*   $T$ is the sample size.

**Real-world application:** Used to determine if the challenger model's NPV predictions are statistically significantly different (and ideally better) than the champion model's predictions.

### 2.6 Kolmogorov-Smirnov Test (KS Test)
The Kolmogorov-Smirnov test (KS test) is a non-parametric test that determines if two samples are drawn from the same distribution. The null hypothesis is that the two samples are drawn from the same distribution.

## 3. Code Requirements

### 3.1 Expected Libraries

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical computations.
*   `joblib`: For loading the champion model.
*   `matplotlib`: For creating static, interactive, and animated visualizations in Python.
*   `seaborn`: For making statistical graphics in Python.
*   `scipy`: For scientific computing, including statistical tests.
*   `statsmodels`: For statistical modeling and econometrics, including the Diebold-Mariano test.
*   `scikit-learn`: For machine learning tasks, if necessary for stress-testing or scenario generation.
*   `pytest`: For unit testing.
*   `jupyter-nbconvert`: For converting the notebook to other formats, such as HTML or PDF.
*   `streamlit`: For deployment using an app.

### 3.2 Input/Output Expectations

*   **Inputs:**
    *   `champion_model.joblib`: Champion model.
    *   `taiwan_realized_cashflows.csv`: Realized cash flow data.
    *   `taiwan_yield_curve_history.csv`: Historical yield curve data.
    *   `taiwan_macro_quarterly.csv`: Macroeconomic scenario data.
    *   `modelrisk_config.yml`: Configuration file containing thresholds and settings.
*   **Outputs:**
    *   `npv_challenger_<run_id>.csv`: NPV results from the challenger model.
    *   `validation_audit_<run_id>.jsonl`: Validation audit log in JSON Lines format.
    *   `issues_<run_id>.csv`: CSV file containing a list of identified issues.
    *   Figures (plots, charts) saved to `figures_part2/`.
    *   Model validation report (format determined later).

### 3.3 Algorithms and Functions to be Implemented

1.  **`Validator` Class:** Orchestrates the entire validation pipeline.
    *   `input_checks()`: Performs data quality checks on input datasets (e.g., missing values, data type validation).
    *   `run_challenger()`: Executes the challenger NPV model and saves results to `npv_challenger_<run_id>.csv`.
    *   `perform_statistical_tests()`: Calculates RMSE/MAE on ΔNPV and performs Diebold-Mariano tests and KS-test.
    *   `perform_stress_tests()`: Executes stress tests based on macroeconomic scenarios and calculates ΔNPV.
    *   `apply_governance_triggers()`: Checks for threshold breaches based on `modelrisk_config.yml`.
    *   `generate_report()`: Generates the model validation report.
    *   `track_issues()`: Records any issues found during the validation, like threshold breaches in a csv file: `issues_<run_id>.csv`.

2.  **NPV Calculation Function (Challenger Model):**
    *   Input: Loan parameters (e.g., principal, interest rate, maturity), cash flow schedule, and discount rate.
    *   Output: NPV of the loan.
    *   Should replicate the champion model's logic independently.

3.  **Back-testing Functions:**
    *   Function to reconstruct discount factors from historical yield curves.
    *   Function to compare predicted NPV with realized cash flows and calculate ΔNPV.

4.  **Statistical Test Functions:**
    *   Function to calculate RMSE and MAE on ΔNPV.
    *   Function to perform the Diebold-Mariano test.
    *   Function to perform Kolmogorov-Smirnov Test

5.  **Stress-Testing Functions:**
    *   Function to simulate stressed scenarios based on macroeconomic variables.
    *   Function to calculate the contribution of each shock to ΔNPV change.

6.  **Governance Functions:**
    *   Function to check for threshold breaches based on `modelrisk_config.yml`.
    *   Function to track and log issues.

7.  **Data Quality Functions**
    *   Function to check for input drift in key features.

### 3.4 Visualizations

*   **Back-test plot:** Scatter plot (or line plot) of predicted vs. realized NPV per loan with a 45° reference line.
*   **Residual control chart:** ΔNPV prediction error over time with ±3σ bands.
*   **Stress-test waterfall chart:** Contribution of each shock (±100 bp, recovery lag, etc.) to ΔNPV change.
*   **Input-drift heat-map:** Monthly drift of key inputs (EIR, tenor) to flag data-quality issues.
*   **Governance dashboard tiles:** Count and severity of threshold breaches, open issues list.

## 4. Additional Notes or Instructions

*   **Assumptions:**
    *   The champion model is assumed to be accurate and reliable as a baseline.
    *   Synthetic datasets are representative of real-world loan portfolios.
*   **Constraints:**
    *   The simulator should be computationally efficient to allow for rapid testing and analysis.
    *   The code should be well-documented and easy to understand.
*   **Customization:**
    *   Allow users to customize stress-test scenarios and governance thresholds via the `modelrisk_config.yml` file.
    *   Provide options to select different statistical tests and visualizations.
*   Fix `random_state=42` in challenger simulations to ensure reproducibility.
*   Store all configurations under `/configs_part2/`.
*   Increment `model_version` on any update; log to `model_inventory.csv`; trigger re-validation automatically.


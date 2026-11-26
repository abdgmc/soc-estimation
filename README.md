# SOC Estimation Project

This student project compares Coulomb counting and an Extended Kalman Filter (EKF) for estimating the state of charge (SOC) of a lithium-ion battery using CALCE INR 18650-20R data.

## Repository contents

- `src/01_ocv_soc_curve.py` — builds an OCV–SOC curve from the low-current test.
- `src/part2_model_fit_validation.py` — prepares the data, fits the Thevenin model, and validates its voltage predictions.
- `src/part3_coulomb_counting_ekf.py` — runs the Coulomb-counting and EKF comparisons.
- `data/model_parameters.json` — model parameters used by Part 3.
- `results/part3_results.csv` — SOC estimation results.
- `results/` — generated plots.

## Data

The raw CALCE Excel files are not included in this repository. Download the low-current OCV, DST 80%, DST 50%, FUDS 80%, and FUDS 50% files from the [CALCE Battery Data page](https://calce.umd.edu/battery-data). Follow CALCE’s terms for using and sharing the data.

## Run in Google Colab

1. Download the Part 2 and Part 3 `.py` scripts from `src/`.
2. Create a new notebook in [Google Colab](https://colab.research.google.com/).
3. In a code cell, install the required packages:

   ```text
   !pip install numpy pandas matplotlib scipy openpyxl xlrd

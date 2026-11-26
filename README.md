# SOC Estimation Project

This repository contains files for a student project on estimating the state of charge (SOC) of a lithium-ion battery using CALCE INR 18650-20R data.

## Current contents

- `src/01_ocv_soc_curve.py` builds an OCV–SOC curve from the CALCE low-current test spreadsheet.
- `data/model_parameters.json` contains the model parameters and summary statistics currently available.

The model-fitting and validation script, Coulomb-counting and EKF experiment script, and `part3_results.csv` referenced in the project report are not currently included.

## Data

The raw CALCE Excel files are not included. Download the required low-current OCV spreadsheet from the [CALCE Battery Data page](https://calce.umd.edu/battery-data) and place it in the `data` folder with the filename expected by the script:

`11_5_2015_low current OCV test_SP20-1.xlsx`

Please follow CALCE’s terms for using and sharing its data.

## Run the OCV–SOC script

Install the required Python packages:

```text
python -m pip install numpy pandas matplotlib openpyxl
```

From the repository’s main folder, run:

```text
python src/01_ocv_soc_curve.py
```

The script saves the OCV–SOC curve data, plot, and polynomial coefficients in the folder where it is run.

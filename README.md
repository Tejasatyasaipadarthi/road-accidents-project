# Road Accident Severity Prediction

Streamlit application for predicting road accident severity with a trained
Random Forest pipeline.

## Project files

Keep these folders in the same repository as `app.py`:

- `data/accidents.csv`
- `models/random_forest_model.pkl`
- `outputs/reports/model_metadata.json`
- `outputs/reports/classification_report.json`

The original target column is `Accident_Severity_C`. The saved model pipeline
includes the preprocessing used during training, and the feature order is
recorded in `model_metadata.json`.

## Run locally

From this directory, install the dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Push the complete contents of this directory to the GitHub repository,
including `data/`, `models/`, and `outputs/reports/`. In Streamlit Community
Cloud, select that repository and branch, set the main file to `app.py`, and
deploy. Do not select only the Python source files or omit the model artifact.

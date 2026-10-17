# Predictive Maintenance with KNN on Azure Machine Learning

**Student:** Zeenat Riaz  
**Roll Number:** L1F22BSAF0157

## Problem

This project predicts machine failure using sensor readings from the AI4I 2020 Predictive Maintenance dataset. The dataset contains numerical sensor measurements and a machine type category. The failure class is imbalanced, so recall for failures is an important evaluation metric.

Dataset: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.

## Approaches

1. **Notebook (scikit-learn + MLflow):** `notebooks/01_knn_notebook.ipynb`
2. **Automated ML (KNN):** `automl/automl_results.md`
3. **Designer:** `designer/knn_designer_script.py`

Data preparation notebook: `notebooks/00_prepare_data.ipynb`

## Results

| Metric | Notebook | Automated ML | Designer |
|---|---:|---:|---:|
| Best K | 47 | 62 | 1 |
| Weights | uniform | uniform | uniform |
| Scaler | RobustScaler | RobustScaler | StandardScaler |
| Failure recall | 0.353 | 0.1476* | 0.3529 |
| Failure F1 | 0.397 | 0.2564* | 0.3967 |
| AUC | 0.669 (test) | 0.94104 (weighted) | 0.6690 |
| Code written | Yes | Minimal | Yes |
| Managed endpoint deployable | Yes | Yes | No (classic pipeline) |

*The AutoML failure recall and F1 values require confirmation against the same test set. The AutoML weighted AUC and the notebook/designer test AUC may use different evaluation methods, so they should not be interpreted as a direct comparison.*

## Comparison Questions

### 1. Which approach gave the best failure recall?

The Notebook and Designer models both achieved approximately 0.353 failure recall, while the recorded AutoML recall was 0.1476 and still needs verification. The notebook and Designer results are very similar on the reported test metrics.

### 2. Which approach would be suitable for a real factory?

Automated ML is useful for quickly exploring model configurations, while a notebook provides more control over preprocessing, tuning, evaluation, and reproducibility. For a real factory, I would validate the model on representative production data, compare failure recall and false alarms, monitor performance, and test the deployment before relying on predictions for maintenance decisions.

### 3. Is a 3.4% failure rate a challenge for KNN?

Yes. A low failure rate creates class imbalance, so a model can achieve high accuracy while missing many actual failures. Possible improvements include collecting more failure examples, adjusting the decision threshold, evaluating class-sensitive approaches, and comparing models that support class weighting. Recall and precision should both be monitored.

## Deployment

A managed online endpoint was used to test the trained model.

- Deployment name: `blue`
- Endpoint name: `knn-maint-raheel`
- Instance type: `Standard_DS2_v2`
- Sample request: `deployment/sample-request.json`
- Test script: `deployment/test_endpoint.py`

The endpoint test returned HTTP 200 with predictions `[0, 0, 0]`. The endpoint has since been deleted as part of cleanup.

## Screenshots

Project screenshots should be stored in the `screenshots/` folder, named by lab step (for example, `1.4_workspace_overview.png` and `8.2_endpoint_prediction.png`).

## What I Learned

This project helped me compare KNN models built with a notebook, Automated ML, and Designer in Azure Machine Learning. I learned how scaling, hyperparameter tuning, and class imbalance affect predictive maintenance results. I also practiced evaluating failure recall and F1 rather than relying on accuracy alone. Finally, I tested a managed online endpoint and documented the deployment and cleanup process.

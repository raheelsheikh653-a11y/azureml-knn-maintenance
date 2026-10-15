# Automated ML results (KNN only)

- Job name: knn-automl-raheel-1
- Best algorithm (scaler + model): RobustScaler + KNN
- n_neighbors: 62
- weights: uniform
- AUC weighted: 0.94104
- Test recall (failure class): 0.1476 (verify against test-set metrics)
- Test F1 (failure class): 0.2564 (verify against test-set metrics)
- Observation: AutoML selected RobustScaler + KNN with n_neighbors=62, weights=uniform, and metric=manhattan. The model achieved a weighted AUC of 0.94104, slightly higher than the notebook KNN model's 0.93971. Although VotingEnsemble appeared as the overall top row, the lab required KNN only, so the best individual KNN trial was selected.
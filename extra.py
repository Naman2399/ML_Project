from sklearn.metrics import roc_auc_score
import numpy as np

# Example data
y_true = np.array([0, 1, 2, 2, 1])  # Example true labels
y_scores = np.array([[0.7, 0.2, 0.1],  # Example predicted probabilities, sum up to 1.0
                     [0.3, 0.5, 0.2],
                     [0.2, 0.3, 0.5],
                     [0.1, 0.2, 0.7],
                     [0.4, 0.3, 0.3]])

# Compute ROC-AUC score
roc_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')  # Using 'ovr' strategy

print(f'ROC-AUC Score: {roc_auc:.2f}')

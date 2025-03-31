from typing import Optional, Callable, List, Tuple

import numpy as np

class EnsembleForecaster:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def ensemble_forecast(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> float:
        """
        Computes a weighted ensemble prediction for a single true value.
        
        Parameters:
            predictions: Array of predictions for a given time step.
            uncertainties: Array of uncertainty values corresponding to each prediction.
        
        Returns:
            float: The weighted ensemble prediction.
        """
        # Compute weights as the inverse of the uncertainty.
        weights = 1 / (uncertainties + self.epsilon)
        weighted_prediction = np.sum(weights * predictions) / np.sum(weights)
        return weighted_prediction

    @staticmethod
    def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def ensemble_loss(
        self,
        predictions_list: List[np.ndarray],
        uncertainties_list: List[np.ndarray],
        true_values: np.ndarray,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Computes ensemble predictions and loss over a series of time steps,
        accommodating a variable number of predictions per time step.
        
        Parameters:
            predictions_list: A list of np.array, where each element contains the predictions
                                     for a given time step.
            uncertainties_list: A list of np.array, where each element contains the uncertainty
                                       values corresponding to the predictions.
            true_values: An array of true observations for each time step.
            loss_fn: A loss function to compare predictions with true values.
                                Defaults to mse_loss.
        
        Returns:
            tuple: (ensemble_predictions, loss_value)
        """
        if loss_fn is None:
            loss_fn = self.mse_loss

        T = len(true_values)
        ensemble_preds = np.empty(T)

        # Compute ensemble prediction for each time step
        for t in range(T):
            preds_t = predictions_list[t]
            uncert_t = uncertainties_list[t]

            if preds_t.size == 0:
                ensemble_preds[t] = np.nan
            else:
                ensemble_preds[t] = self.ensemble_forecast(preds_t, uncert_t)

        # Filter out time steps where ensemble prediction is not computed
        valid_indices = ~np.isnan(ensemble_preds)
        loss_value = loss_fn(true_values[valid_indices], ensemble_preds[valid_indices])
        return ensemble_preds, loss_value

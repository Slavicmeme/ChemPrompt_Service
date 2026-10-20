# models/base_model.py

class BaseModel:
    def __init__(self):
        """
        BaseModel is an abstract base class that all model classes should inherit from.
        Common initialization or structure can be defined here.
        """
        pass

    def fit_and_evaluate(self, data_list, x, y):
        """
        Method to train and evaluate the model.
        Must be implemented in the child class.
        """
        raise NotImplementedError("The method fit_and_evaluate() must be implemented in the child class.")

    def _save_predictions(self, predictions):
        """
        Method to save prediction results.
        Should be implemented in the child class.
        """
        raise NotImplementedError("The method _save_predictions() must be implemented in the child class.")

    def _save_combined_results(self, metric_results):
        """
        Method to save evaluation metric results.
        Should be implemented in the child class.
        """
        raise NotImplementedError("The method _save_combined_results() must be implemented in the child class.")

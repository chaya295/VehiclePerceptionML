# Inference Pipeline for Predictions

import numpy as np
import joblib

class InferencePipeline:
    def __init__(self, model_path):
        # Load the trained model
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        """Make predictions based on the input data."""
        # Ensure the input data is in the right format
        input_array = np.array(input_data).reshape(1, -1)  # Change as per your model's requirements
        predictions = self.model.predict(input_array)
        return predictions

    def predict_proba(self, input_data):
        """Predict class probabilities using the trained model."""
        input_array = np.array(input_data).reshape(1, -1)
        probabilities = self.model.predict_proba(input_array)
        return probabilities

# Example usage:
# pipeline = InferencePipeline('path_to_model.pkl')
# prediction = pipeline.predict(input_data)
# print('Predicted class:', prediction)
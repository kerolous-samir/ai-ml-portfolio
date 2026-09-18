from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the trained model
MODEL_PATH = "superkart_best_model.pkl"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.get("/")
def home():
    return "Hello to SuperKart API Model!!"

@app.post("/predict")
def predict():
    try:
        # Get JSON body
        input_json = request.json
        if input_json is None:
            return jsonify({"error": "No JSON body found"}), 400

        # Accept either a single dict or a list of dicts
        if isinstance(input_json, dict):
            input_data = [input_json]
        elif isinstance(input_json, list):
            input_data = input_json
        else:
            return jsonify({
                "error": "Invalid input format. Provide a JSON object or a list of JSON objects."
            }), 400

        # Convert to DataFrame
        df = pd.DataFrame(input_data)

        # -------- Feature engineering (must match training pipeline) --------
        # Product_Category from first 2 chars of Product_Id
        if 'Product_Id' in df.columns:
            df['Product_Category'] = df['Product_Id'].str[:2]

        # Store_Age from Store_Establishment_Year.
        # 2009 is the max establishment year in the TRAINING set, frozen here on
        # purpose. Training computed it as Store_Establishment_Year.max(); at serving
        # time a request may hold a single row, so recomputing .max() would yield
        # Store_Age == 0 every time and silently skew predictions away from training.
        if 'Store_Establishment_Year' in df.columns:
            df['Store_Age'] = 2009 - df['Store_Establishment_Year']

        # MRP_per_weight
        if {'Product_MRP', 'Product_Weight'}.issubset(df.columns):
            df['MRP_per_weight'] = df['Product_MRP'] / df['Product_Weight'].replace(0, np.nan)

        # Drop ID-like columns not used for training
        for col in ['Product_Id', 'Store_Id']:
            if col in df.columns:
                df = df.drop(columns=col)

        # Predict
        preds = model.predict(df)

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)

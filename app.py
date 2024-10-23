from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib  # Import joblib to load the scaler

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler
scaler = joblib.load("scaler.pkl")  # Make sure the scaler is in the correct path


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input features from the request (expects JSON input)
        data = request.json
        print(data)  # Log the incoming data for debugging

        # Extract relevant input features
        gender = data.get("gender", None)
        patient_age = int(data.get("patientAge", 0))
        diagnostic_dm_years = float(data.get("diagnosticDmYears", 0))
        smoking = data.get("smoking", None)
        alcohol = data.get("alcohol", None)
        bmi = float(data.get("bmi", 0))
        lying_sbp_avg = float(data.get("lyingSbpAvg", 0))
        stand_3min_sbp = float(data.get("stand3MinSbp", 0))
        lying_dbp_avg = float(data.get("lyingDbpAvg", 0))
        stand_3min_dbp = float(data.get("stand3MinDbp", 0))
        ht_status = data.get("htStatus", None)
        hr_bpm = float(data.get("hrBpm", 0))
        waist_circumference = float(data.get("waistCircumference", 0))
        fasting_glucose = float(data.get("fastingGlucose", 0))
        hba1c = float(data.get("hba1c", 0))
        urea = float(data.get("urea", 0))
        creatinine = float(data.get("creatinine", 0))
        triglyceride = float(data.get("triglyceride", 0))
        hdl = float(data.get("hdl", 0))
        ldl = float(data.get("ldl", 0))
        cvd_risk = float(data.get("cvdRisk", 0))
        crp = float(data.get("crp", 0))
        gfr = float(data.get("gfr", 0))
        ewing_result = int(data.get("ewingResult", 0))

        # Prepare input for the model as a NumPy array
        features = np.array(
            [
                [
                    1 if gender == "Male" else 0,  # Example: encoding for gender
                    patient_age,
                    diagnostic_dm_years,
                    1 if smoking == "Yes" else 0,  # Example: encoding for smoking
                    1 if alcohol == "Yes" else 0,  # Example: encoding for alcohol
                    bmi,
                    lying_sbp_avg,
                    stand_3min_sbp,
                    lying_dbp_avg,
                    stand_3min_dbp,
                    1 if ht_status == "Yes" else 0,  # Example: encoding for HT Status
                    hr_bpm,
                    waist_circumference,
                    fasting_glucose,
                    hba1c,
                    urea,
                    creatinine,
                    triglyceride,
                    hdl,
                    ldl,
                    cvd_risk,
                    crp,
                    gfr,
                    ewing_result,
                ]
            ]
        )

        print("Raw Features:", features)  # Log the raw features for debugging

        # Scale the input features using the loaded scaler
        input_features_scaled = scaler.transform(features)

        # Set the input tensor to the scaled features
        interpreter.set_tensor(
            input_details[0]["index"], input_features_scaled.astype(np.float32)
        )

        # Run the interpreter
        interpreter.invoke()

        # Get the prediction result
        prediction_probs = interpreter.get_tensor(output_details[0]["index"])
        predicted_class = (prediction_probs > 0.5).astype(int)[0][
            0
        ]  # Binary classification
        predicted_label = "Diabetic" if predicted_class == 1 else "Non-Diabetic"

        # Return the prediction as a JSON response
        return jsonify(
            {
                "Predicted Class": int(predicted_class),
                "Predicted Label": predicted_label,
                "Prediction Probabilities": prediction_probs.tolist(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)

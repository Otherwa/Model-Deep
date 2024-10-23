from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler (ensure you save and load your scaler properly)
scaler = joblib.load("scaler.pkl")  # Load your pre-fitted scaler


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input features from the request (expects JSON input)
        data = request.json

        # Extract relevant input features
        gender = data.get(
            "Gender", None
        )  # Gender may need encoding if used in the model
        patient_age = int(data.get("Patient Age", 0))
        diagnostic_dm_years = float(data.get("Diagnostic DM (years)", 0))
        smoking = data.get("Smoking?", None)  # Encode if necessary
        alcohol = data.get("Alcohol?", None)  # Encode if necessary
        bmi = float(data.get("BMI", 0))
        lying_sbp_avg = float(data.get("Lying SBP average", 0))
        stand_3min_sbp = float(data.get("Stand 3min SBP", 0))
        lying_dbp_avg = float(data.get("Lying DBP average", 0))
        stand_3min_dbp = float(data.get("Stand 3min DBP", 0))
        ht_status = data.get("HT Status", None)  # Encode if necessary
        hr_bpm = float(data.get("HR (BPM) (t_biochemistry_C16)", 0))
        waist_circumference = float(data.get("Waist Circumference", 0))
        fasting_glucose = float(data.get("Fasting Glucose(mmol/L)", 0))
        hba1c = float(data.get("HbA1c (%)", 0))
        urea = float(data.get("Urea(mmol/L)", 0))
        creatinine = float(data.get("Creatinine(mmol/L)", 0))
        triglyceride = float(data.get("Triglyceride(mmol/L)", 0))
        hdl = float(data.get("HDL(mmol/L)", 0))
        ldl = float(data.get("LDL(mmol/L)", 0))
        cvd_risk = float(data.get("CVD % risk 5 years", 0))
        crp = float(data.get("CRP", 0))
        gfr = float(data.get("GFR", 0))
        ewing_result = int(data.get("Ewing Result", 0))

        # Prepare input for the model as a NumPy array
        features = np.array(
            [
                [
                    # You may need to encode categorical variables
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

        # Scale the input features
        features_scaled = scaler.transform(features)

        # Set the input tensor to the scaled features
        interpreter.set_tensor(
            input_details[0]["index"], features_scaled.astype(np.float32)
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

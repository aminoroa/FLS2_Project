import sys
import joblib
import pandas as pd


def main():
    # Path to your fully trained and saved model
    model_path = (
        "/Users/aminoroa/FLS2_Project/ml_data/ML_run_final_logistic_model.joblib"
    )

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- INPUT YOUR NEW DATA HERE ---
    # Replace these placeholder values with the actual scores for your new complex.
    # If you ran 3 replicates, input the average for each score here.
    new_scores = {
        "iptm": [0.88],
        "ptm": [0.85],
        "ligand_fls2_iptm": [0.92],
        "ligand_coreceptor_iptm": [0.75],
        "ligand_pae_min_fls2": [2.1],
        "ligand_pae_min_coreceptor": [5.4],
    }

    # Convert the dictionary to a Pandas DataFrame.
    # This is important so the model recognizes the exact feature names it was trained on.
    df_new = pd.DataFrame(new_scores)

    # Make the prediction
    prediction = model.predict(df_new)

    # Calculate the probability/confidence for each class
    probabilities = model.predict_proba(df_new)

    print("\n========================================")
    print("          PREDICTION RESULTS            ")
    print("========================================")
    print(f"Predicted Class: >> {prediction[0].upper()} <<\n")

    print("Model Confidence Breakdown:")
    # model.classes_ stores the class names in the exact order the probabilities are output
    for cls, prob in zip(model.classes_, probabilities[0]):
        print(f"  - {cls}: {prob:.2%}")
    print("========================================\n")


if __name__ == "__main__":
    main()

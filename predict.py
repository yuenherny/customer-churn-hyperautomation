import os
import sys

import pandas as pd
import pickle as pkl
import time

INPUT_PATH = "input"
FILE_NAME = sys.argv[1]

OUTPUT_PATH = "output"
MODELS_PATH = "models"

def main():
    df = pd.read_csv(os.path.join(INPUT_PATH, FILE_NAME))

    # Transformation
    # Remove features
    df2 = df.copy()
    col_remove = ["gender", "PhoneService", "MultipleLines", "customerID"]

    df2 = df2.drop(col_remove, axis=1)

    # Convert text to binary
    col_text2binary = ["SeniorCitizen", "Partner", "Dependents", 
                        "PaperlessBilling"]

    for col in col_text2binary:
        df2[col] = df2[col].apply(lambda x: 1 if x == "Yes" else 0)

    # One hot encoding
    with open(os.path.join(MODELS_PATH, "onehotenc.pkl"), "rb") as f:
        enc = pkl.load(f)

    col_text2onehot = ["InternetService", "OnlineSecurity", "OnlineBackup", 
                        "DeviceProtection", "TechSupport", "StreamingTV", 
                        "StreamingMovies", "Contract", "PaymentMethod"]

    onehot_output = enc.fit_transform(df2[col_text2onehot]).toarray()
    df3 = pd.DataFrame(data=onehot_output, columns=enc.get_feature_names_out(col_text2onehot))

    # Combine binary with one hot
    X_test = pd.concat([df2[col_text2binary], df3], axis=1)

    # Perform inference using Sklearn Pipeline
    with open(os.path.join(MODELS_PATH, "logr.pkl"), "rb") as f:
        pipe_logr = pkl.load(f)

    y_pred = pipe_logr.predict(X_test.values)
    df_test = pd.DataFrame(data=y_pred, columns=["Churn"])

    df = pd.concat([df, df_test], axis=1)

    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)

    SAVED_FILEPATH = os.path.join(OUTPUT_PATH, f"output_{current_time}.csv")
    df.to_csv(SAVED_FILEPATH, index=False)

    print(SAVED_FILEPATH)

if __name__ == "__main__":
    main()
from flask import Flask, jsonify, request
import pandas as pd
import pickle
import os


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def generate_predictions():
    input = request.json
    df = pd.DataFrame(data=input)
    model_pickle_path = "churn_prediction_model.pkl"
    label_encoder_pickle_path = "churn_prediction_label_encoders.pkl"

    model, label_encoder_dict = load_pickles(model_pickle_path,
                                             label_encoder_pickle_path)

    processed_df = pre_process_data(df, label_encoder_dict)
    prediction = make_predictions(processed_df, model)
    print(prediction)

def load_pickles(model_pickle_path, label_encoder_pickle_path):
    model_pickle_opener = open(model_pickle_path,"rb")
    model = pickle.load(model_pickle_opener)

    label_encoder_pickle_opener = open(label_encoder_pickle_path,"rb")
    label_encoder_dict = pickle.load(label_encoder_pickle_opener)

    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df.drop("customerID", axis=1, inplace=True)
    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            column_le = label_encoder_dict[col]
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue
    # TODO: add assert statement that all cols are numeric
    return df


def make_predictions(processed_df, model):
    prediction = model.predict(processed_df)
    return prediction

if __name__ == "__main__":
    app.run()

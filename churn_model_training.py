import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
holdout = df.iloc[-100:, :]
holdout.drop("Churn", axis=1, inplace=True)
holdout.to_json("holdout_test.json", orient="records")
df.drop("customerID", axis=1, inplace=True)


# train data
train = df.iloc[:-100, :]

# holding out last 100 rows to use later for predicting with flask app
holdout = df.iloc[-100:, :]

"""
Data Pre-Processing
The following columns need to be fixed in order to train the model:
1. All yes/no categories category encoded
2. All category columns category encoded
3. "Churn" column category encoded
"""
categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn']
# converting all the categorical columns to numeric
col_mapper = {}
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(train.loc[:, col])
    class_names = le.classes_
    train.loc[:, col] = le.transform(train.loc[:, col])
    # saving encoder for each column to be able to inverse-transform later
    col_mapper.update({col: le})

train.replace(" ", "0", inplace=True)

# converting "Total Charges" to numeric
train.loc[:, "TotalCharges"] = pd.to_numeric(train.loc[:, "TotalCharges"])

# splitting into X and Y
x_train = train.drop("Churn", axis=1)
y_train = train.loc[:, "Churn"]


model = LogisticRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_train)
precision, recall, fscore, support = precision_recall_fscore_support(y_train, predictions)
accuracy = accuracy_score(y_train, predictions)
accuracy


# pickling mdl
pickler = open("churn_prediction_model.pkl", "wb")
pickle.dump(model, pickler)
pickler.close()

# pickling le dict
pickler = open("churn_prediction_label_encoders.pkl", "wb")
pickle.dump(col_mapper, pickler)
pickler.close()

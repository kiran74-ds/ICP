# pylint: disable=missing-module-docstring,import-error

import logging
import pickle
from logging.config import dictConfig
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import prepare_data, train_model


dictConfig({'version': 1, 'root': {'level': 'INFO'}})
logger = logging.getLogger('Classification')
logging.basicConfig(level=logging.INFO)

def prepare_data_and_train_model():
    # pylint: disable=invalid-name
    '''
    Prepare and train the ML model on input data
    :return: model and test data
    '''
    logging.info("prepare data and train model")
    data_frame = pd.read_csv("./app/src/data/train.csv", index_col=[0])
    data_frame = prepare_data(data_frame)
    print(data_frame.columns)
    data_frame = data_frame.dropna(axis='columns')
    target = data_frame["ICP-conform"]
    features = data_frame.drop("ICP-conform", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target)
    pd.Series(X_train.columns.values.tolist(), name='features').to_csv('./app/src/data/features.csv')
    print(y_train)
    print("Input features during training ", len(X_train.columns))
    rf_model, rf_model_accuracy = train_model(RandomForestClassifier, X_train, y_train)
    pickle.dump(rf_model, open('model_weights', 'wb'))
    return rf_model, rf_model_accuracy, X_test, y_test


def get_predictions():
    # pylint: disable=invalid-name
    '''
    make predictions on the test data
    :return: predictions
    '''
    logging.info("Getting Predictions on unseen data")
    test_data = pd.read_csv("./app/src/data/test.csv")
    loaded_model = pickle.load(open('model_weights', 'rb'))
    X_test = prepare_data(test_data)
    predictions = loaded_model.predict(X_test)
    logging.info(predictions)
    test_data['predictions'] = predictions
    logging.info("Test data Predictions")
    logging.info(test_data[['ICP-conform', 'predictions']])
    return predictions


if __name__ == "__main__":
    model, model_accuracy, test_features, test_label  = prepare_data_and_train_model()
    # logging.info("Getting Predictions on test data")
    # test_predictions = get_predictions()


import logging
import sys
import pickle
import pandas as pd
from logging.config import dictConfig
from preprocess import prepare_data, train_model
from sklearn.metrics import classification_report

dictConfig({'version': 1, 'root': {'level': 'INFO'}})
logger = logging.getLogger('Classification')
logging.basicConfig(level=logging.INFO)

def get_predictions(test_file_path, label=False):
    # pylint: disable=invalid-name
    '''
    make predictions on the test data
    :return: predictions
    '''
    logging.info("Getting Predictions on unseen data")
    
    test_data = pd.read_csv(test_file_path)
    loaded_model = pickle.load(open('model_weights', 'rb'))
    X_test = prepare_data(test_data, 'test')
    predictions = loaded_model.predict(X_test)
    logging.info(predictions)
    test_data['predictions'] = predictions
    if label==True:
        target = test_data['ICP-conform']
        accuracy_score = round(loaded_model.score(X_test, target) * 100, 2)
        print(f'accuracy ({loaded_model.__repr__()}): {accuracy_score}')
        logging.info(classification_report(target, predictions))

    return predictions


if __name__ == "__main__":
    logging.info(sys.argv[0])
    #test_file_path = "./app/src/data/test.csv"
    test_file_path = sys.argv[1]
    if len(sys.argv)>2:
        label = sys.argv[2]
    else:
        label = True
    test_predictions = get_predictions(test_file_path, label)
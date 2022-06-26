# pylint: disable=missing-module-docstring
import logging
import pandas as pd
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

def one_hot_coding(data_frame, columns):
    '''Creating Dummy Columns
    args:
        data_frame : data frame
        columns: list of column names
    return -- dataframe
        '''
    logging.info("Creating Dummy Varibles")
    for column in columns:
        data_frame = pd.concat([data_frame, pd.get_dummies(data_frame[column],
                                                           prefix=column)], axis=1)
        data_frame.drop(column, inplace=True, axis=1)

    return data_frame

def train_model(model_class, features, target):
    '''
    :param model_class: Machine Learning Classification Algorithm
    :param features: Input features
    :param target: target
    :return: model, model accuracy
    '''

    logging.info("Training Model")

    model = model_class()
    model.fit(features, target)
    predictions = model.predict(features)
    accuracy_score = round(model.score(features, target) * 100, 2)
    print(f'accuracy ({model.__repr__()}): {accuracy_score}')
    logging.info(classification_report(target, predictions))

    return model, accuracy_score

def prepare_data(df, preparation_type=None):
    try:
        logging.info("Preparing Data")
        df = df[['companyType', 'employeesCount', 'timestamp', 'ICP-conform']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month_num'] = df['timestamp'].dt.month
        df['month'] = df['timestamp'].dt.month_name()
        df['day_num'] = df['timestamp'].dt.dayofweek
        df['day'] = df['timestamp'].dt.day_name()
        df['hour'] = df['timestamp'].dt.hour
        features_df = one_hot_coding(df[['companyType', 'employeesCount', 'day', 'month', 'hour', 'ICP-conform']],
                                         ['companyType', 'employeesCount', 'day', 'month', 'hour'])
        if preparation_type=='test':
            logging.info("Preparing Data for test dataset")
            features = pd.read_csv('./app/src/data/features.csv')['features'].values.tolist()
            logging.info(len(features))
            new_features =  set(df.columns.tolist()) - set(features)
            missing_features = list(set(features) - set(df.columns.tolist()))
            features_df[missing_features] = 0
            features_df = features_df[features]
        
        print(features_df.shape)
        return features_df  
    
    except Exception as e:
        print(e)

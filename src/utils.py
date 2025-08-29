###common functionalitites that the entire project can use

import os
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            logging.info(f"Training the {model_name}")
            model.fit(X_train, y_train)

            logging.info(f"Evaluating the {model_name}")
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"{model_name} has a score of {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

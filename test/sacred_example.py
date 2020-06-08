# encoding: utf-8
from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ex = Experiment("linear 5")
obv = MongoObserver(url="localhost", port=8888, db_name="ml")
ex.observers.append(obv)

@ex.config
def config():
    alpha = 0.1

@ex.automain
def run():
    x = np.random.rand(200, 8)
    y = np.random.rand(200)
    model = Ridge(alpha=1.2)
    model.fit(x, y)
    mse = mean_squared_error(model.predict(x), y)
    ex.log_scalar("mse", mse)
    return float(mse)

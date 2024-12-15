# -*- coding: cp1251 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, balanced_accuracy_score
from regression import *
from collections import Counter


data = pd.read_csv("organisations.csv")
features = pd.read_csv("features.csv")
rubrics = pd.read_csv("rubrics.csv")

rubric_dict = {}
for i in range(rubrics["rubric_id"].size):
    rubric_dict[rubrics["rubric_id"][i]] = rubrics["rubric_name"][i]

data = data.dropna(subset=["average_bill"])
data = data[data["average_bill"] <= 2500]

data_train, data_test = train_test_split(
    data, stratify=data["average_bill"], test_size=0.33, random_state=42
)


# reg = MeanRegressor()
# reg.fit(y=data_train["average_bill"])

# clf = MostFrequentClassifier()
# clf.fit(y=data_train["average_bill"])

# rmse_res = root_mean_squared_error(
#     data_test["average_bill"], np.full(data_test["average_bill"].size, reg.predict())
# )
# print(rmse_res)

# bas_res = balanced_accuracy_score(
#     data_test["average_bill"], np.full(data_test["average_bill"].size, clf.predict())
# )
# print(bas_res)


# rmse_res2 = root_mean_squared_error(
#     data_test["average_bill"], np.full(data_test["average_bill"].size, clf.predict())
# )
# print(rmse_res2)

# cmr = CityMeanRegressor()
# cmr.fit(data_train)
# cmr_res = root_mean_squared_error(data_test["average_bill"], cmr.predict(data_test))
# print(cmr_res)

cnt = Counter(data["rubrics_id"])
cnt_clear = [k for k, v in dict(cnt).items() if v >= 100]

data_train["modified_rubrics"] = data_train["rubrics_id"]
data_test["modified_rubrics"] = data_test["rubrics_id"]

scf = SuperClassifier()
scf.fit(X=data_train)
scf_res = scf.predict(data_test)
print(root_mean_squared_error(data_test['average_bill'], scf_res))
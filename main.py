# All required libraries 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

crops = pd.read_csv("soil_measures.csv")
crops.isna().sum()
crops.crop.unique()
features =crops.drop(columns = ["crop"])
target = crops["crop"]
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size = 0.2, random_state=42)
features_dict={}

feature_performance = {}
for feature in ["N","P","K","ph"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[feature].values.reshape(-1, 1), y_train)
    y_pred = log_reg.predict(X_test[feature].values.reshape(-1, 1))
    f1 = f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = f1
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    crop_scores = dict(zip(y_test.unique(), per_class_f1))
    print(f"F1-score for {feature}: {f1}")
    print(f"Per-class F1 scores for {feature}: {crop_scores}")

best_predictive_feature = {"K":feature_performance["K"]}
print(best_predictive_feature)
best_predictive_crop = max(crop_scores, key=crop_scores.get)
print(f"Best predictive crop for feature {best_predictive_feature}: {best_predictive_crop}")

  

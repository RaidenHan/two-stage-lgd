import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from dtreeviz.trees import dtreeviz
import shap
import matplotlib.pyplot as plt

ori_data=pd.read_csv('/Users/zitaosong/Desktop/NCSU/project/two_stage_LGD /Fitting_Data_FullRange_MIOnly.csv', sep=',')
ori_data=ori_data[['LGD','ORIG_RATE','ORIG_UPB',"OLTV",'DTI','CSCORE_B','MI_PCT']]
ori_data=ori_data.dropna()
ori_data=ori_data[ori_data['LGD']!=0]
ori_data=ori_data[ori_data['LGD']!=1]
ori_data.isnull().sum()
ori_data.value_counts()
ori_data.head(20)
x_train,x_test,y_train,y_test=train_test_split(ori_data.drop(['LGD'],axis=1),ori_data['LGD'],
                                               test_size=0.2,random_state=100)

#Random Forest Regression
rfr=RandomForestRegressor(n_estimators=100,max_depth=10,random_state=100)
rfr=rfr.fit(x_train,y_train)
x_train_rfr=x_train.copy()

while True:
    rfr.fit(x_train_rfr,y_train)
    crit=min(0.05,0.5/x_train_rfr.shape[1])
    sel_feat=rfr.feature_importances_>crit
    if sel_feat.all():
        break
    else:
        x_train_rfr=x_train.loc[:,sel_feat]
features=x_train_rfr.columns
score_dict={}
r_dict={}
for max_depth in range(5,15):
    rfr1=RandomForestRegressor(n_estimators=100,max_depth=max_depth, random_state=100)
    rfr1.fit(x_train_rfr,y_train)
    score=mean_squared_error(y_test,rfr1.predict(x_test[features]))
    r=rfr1.score(x_test[features],y_test)
    score_dict[max_depth]=score
    r_dict[max_depth]=r



rfr2=RandomForestRegressor(n_estimators=100,max_depth=9,min_samples_leaf=5,max_features='sqrt',random_state=100)
rfr2.fit(x_train_rfr[features],y_train)




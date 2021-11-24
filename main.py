import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn import feature_selection
from sklearn import model_selection


df = pd.read_excel('/content/gdrive/MyDrive/urban_data.xlsx')

df.keys()
target1 = df['보행환경만족도-거주']
target2 = df['보행환경만족도-시내']

New_target1=[]
for i in target1:
    if i > 3:
        New_target1.append(int(1))
    else:
        New_target1.append(int(0))
new_target1 = np.array(New_target1)

New_target2=[]
for i in target1:
    if i > 3:
        New_target2.append(int(1))
    else:
        New_target2.append(int(0))
new_target2 = np.array(New_target2)

data = df[ ['주거형태', '주거점유형태', '성별', '연령', '학력', '월평균소득', '통근교통수단', '서울시민자부심', '삶 만족도', '지난 2주간 스트레스', '종합위험피해심각도점수','지역대분류-권']]
x_train1, x_test1, y_train1, y_test1 = train_test_split(data, new_target1, test_size=0.3, shuffle=False, random_state=34)
x_train2, x_test2, y_train2, y_test2 = train_test_split(data, new_target2, test_size=0.3, shuffle=False, random_state=34)

###Grid Search Code###

### XGB classifier grid search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

alg = XGBClassifier()
params = {'n_estimators' : [50,100,200], 'learning_rate' : [0.01, 0.05 ,0.1],'max_depth':[1,3,5,7,9], 'min_child_weight':[1,3,5], 'colsample_bytree':[0.5, 0.8, 0.9]}
gridcv = GridSearchCV(alg, param_grid=params, cv=5, refit=True)
gridcv.fit(x_train1, y_train1)
XGB_best = gridcv.best_params_

### adaboost classifier grid search
alg = ensemble.AdaBoostClassifier()
params = {'learning_rate' : [0.1, 0.5, 1.0],'n_estimators':[10, 25,50,100]}
gridcv = GridSearchCV(alg, param_grid=params, cv=5, refit=True)
gridcv.fit(x_train1, y_train1)
ada_best = gridcv.best_params_

### bagging classifier grid search
alg = ensemble.BaggingClassifier()
bc_params = {
          'bootstrap_features': [False, True],
          'max_features': [0.5, 0.7, 1.0],
          'max_samples': [0.5, 0.7, 1.0],
          'n_estimators': [20,50,100,200],
}
gridcv = GridSearchCV(alg, param_grid=bc_params , cv=5, refit=True)
gridcv.fit(x_train1, y_train1)
bagging_best = gridcv.best_params_

### RandomForest classifier grid search
alg = ensemble.RandomForestClassifier()
params = {'max_depth' : [3,5,7,9],'n_estimators':[ 200,300,400],'max_features': ['auto', 'sqrt', 'log2'],'criterion' :['gini', 'entropy']}
gridcv = GridSearchCV(alg, param_grid=params, cv=5, refit=True)
gridcv.fit(x_train1, y_train1)
RF_best = gridcv.best_params_

### Logistic Regression classifier grid search
alg = LogisticRegression()
params = {'C': [0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2', 'elasticnet']}
gridcv = GridSearchCV(alg, param_grid=params, cv=5, refit=True)
gridcv.fit(x_train1, y_train1)
LR_best = gridcv.best_params_

### SVM grid search
alg = svm.LinearSVC()
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2'],
              'loss' : ['hinge','squared_hinge']}
gridcv = GridSearchCV(alg, param_grid=params, cv=5, refit=True)
gridcv.fit(x_train1, y_train1)
SVM_best = gridcv.best_params_

### MLP grid search
alg = MLPClassifier()
params = {'hidden_layer_sizes' : [2,3,4,5,6],
              'activation' : ['relu','tanh'],
              'solver' : ['sgd','adam'],
              'alpha' : [0.001,0.01,0.1],
              'learning_rate' : ['constant','adaptive']}
gridcv = GridSearchCV(alg, param_grid=params, cv=5, refit=True)
gridcv.fit(x_train1, y_train1)
MLP_best = gridcv.best_params_

###Model train and evaluate
models = [
    #svm
    svm.LinearSVC(C= 0.1, loss= 'hinge', penalty= 'l2'),

    #Ensemble
    ensemble.AdaBoostClassifier(learning_rate= 0.1, n_estimators= 50),
    ensemble.BaggingClassifier(bootstrap_features= True, max_features= 0.7, max_samples= 1.0, n_estimators= 200),
    ensemble.RandomForestClassifier(criterion= 'entropy', max_depth= 3, max_features= 'auto', n_estimators= 200),

    #Generalized Linear Models
    LogisticRegression(C= 10, penalty= 'l2'),

    #XGBoost
    XGBClassifier(colsample_bytree= 0.8, learning_rate= 0.05, max_depth= 5, min_child_weight= 3, n_estimators= 50),

    #Neural Network
    MLPClassifier(activation= 'tanh', alpha= 0.1, hidden_layer_sizes= 6, learning_rate= 'adaptive', solver= 'adam'),
    ]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0 )


MLA_columns = ['Algorithm Name', 'Parameters','Train Accuracy Mean', 'Test Accuracy Mean', 'Test Accuracy STD' ,'Learning Time']
MLA_compare_1 = pd.DataFrame(columns = MLA_columns)
MLA_compare_2 = pd.DataFrame(columns = MLA_columns)


row_index = 0
for alg in models:
    MLA_name = alg.__class__.__name__
    MLA_compare_1.loc[row_index, 'Algorithm Name'] = MLA_name
    MLA_compare_1.loc[row_index, 'Parameters'] = str(alg.get_params())
    MLA_compare_2.loc[row_index, 'Algorithm Name'] = MLA_name
    MLA_compare_2.loc[row_index, 'Parameters'] = str(alg.get_params())
    cv_results_1 = model_selection.cross_validate(alg, data, new_target1, scoring='accuracy', cv=cv_split, return_train_score=True)
    cv_results_2 = model_selection.cross_validate(alg, data, new_target2, scoring='accuracy', cv=cv_split, return_train_score=True)
    MLA_compare_1.loc[row_index, 'Learning Time'] = cv_results_1['fit_time'].mean()
    MLA_compare_1.loc[row_index, 'Train Accuracy Mean'] = cv_results_1['train_score'].mean()
    MLA_compare_1.loc[row_index, 'Test Accuracy Mean'] = cv_results_1['test_score'].mean()
    MLA_compare_1.loc[row_index, 'Test Accuracy STD'] = cv_results_1['test_score'].std()
    MLA_compare_2.loc[row_index, 'Learning Time'] = cv_results_2['fit_time'].mean()
    MLA_compare_2.loc[row_index, 'Train Accuracy Mean'] = cv_results_2['train_score'].mean()
    MLA_compare_2.loc[row_index, 'Test Accuracy Mean'] = cv_results_2['test_score'].mean()
    MLA_compare_2.loc[row_index, 'Test Accuracy STD'] = cv_results_2['test_score'].std()
    row_index+=1
MLA_compare_1.sort_values(by = ['Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare_2.sort_values(by = ['Test Accuracy Mean'], ascending = False, inplace = True)

MLA_columns_pred = ['Algorithm Name','precision_score', 'recall_score', 'f1_score']
MLA_result_1 = pd.DataFrame(columns = MLA_columns_pred)
MLA_result_2 = pd.DataFrame(columns = MLA_columns_pred)
from sklearn import datasets, metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
for alg in models:
    MLA_name = alg.__class__.__name__
    cv_predict_1 = model_selection.cross_val_predict(alg, data, new_target1, cv= 5)
    cv_predict_2 = model_selection.cross_val_predict(alg, data, new_target2, cv= 5)
    
    MLA_result_1.loc[row_index, 'Algorithm Name'] = MLA_name
    MLA_result_2.loc[row_index, 'Algorithm Name'] = MLA_name
    MLA_result_1.loc[row_index, 'precision_score'] = precision_score(new_target1,cv_predict_1)
    MLA_result_1.loc[row_index, 'recall_score'] = recall_score(new_target1,cv_predict_1)
    MLA_result_1.loc[row_index, 'f1_score'] = f1_score(new_target1,cv_predict_1)
    MLA_result_2.loc[row_index, 'precision_score'] = precision_score(new_target2,cv_predict_2)
    MLA_result_2.loc[row_index, 'recall_score'] = recall_score(new_target2,cv_predict_2)
    MLA_result_2.loc[row_index, 'f1_score'] = f1_score(new_target2,cv_predict_2)
    row_index+=1
MLA_result_1.sort_values(by = ['f1_score'], ascending = False, inplace = True)
MLA_result_2.sort_values(by = ['f1_score'], ascending = False, inplace = True)

### Result (Accuracy)
print(MLA_compare_1)
print(MLA_compare_2)

## Result (Precision, Recall)
print(MLA_result_1)
print(MLA_result_2)

### Feature Importance ###
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
clf1 =  ensemble.RandomForestClassifier(criterion= 'entropy', max_depth= 3, max_features= 'auto', n_estimators= 200)
clf1.fit(data, new_target1)
clf1.feature_importances_
feature_names = ['주거형태', '주거점유형태', '성별', '연령', '학력', '월평균소득', '통근교통수단', '서울시민자부심', '삶 만족도', '지난 2주간 스트레스', '종합위험피해심각도점수','지역대분류-권'])
clf1.feature_importances_

### ROC curve ###
clf = ensemble.RandomForestClassifier(criterion= 'entropy', max_depth= 3, max_features= 'auto', n_estimators= 200)
clf.fit(data, new_target2)
clf1 = XGBClassifier(colsample_bytree= 0.8, learning_rate= 0.05, max_depth= 5, min_child_weight= 3, n_estimators= 50)
clf1.fit(data, new_target1)
# metrics.plot_roc_curve(clf1, data, new_target1)
metrics.plot_roc_curve(clf, data, new_target2)

### Result Graph ###
sns.barplot(x='Test Accuracy Mean', y = 'Algorithm Name', data = MLA_compare_1, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

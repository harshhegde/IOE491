#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Harsh Hegde


# In[ ]:


#packages imported
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import Ridge
import warnings
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sns.set_style("white")
from sklearn.metrics import matthews_corrcoef

get_ipython().run_line_magic('store', '-r')


# In[ ]:


get_ipython().system('pip install imbalanced-learn')

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
get_ipython().system('pip install category-encoders')


# In[ ]:


from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import SelectKBest

# Models
from sklearn.svm import SVC

# Model Selection functions
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# Metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support as score

# Others
import time
from sklearn.utils import resample
import scipy.interpolate

import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('whitegrid')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = (6, 3)


# # Functions

# In[ ]:


def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):   #function to plot the confusion matrix
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, cmap="YlGnBu", vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    plt.title(title)
    plt.ylabel('True heart disease')
    plt.xlabel('Predicted heart disease')


# In[ ]:


def evaluate_model(model, X_test, y_test):  #function to do the model evaluation
    assert len(X_test) == len(y_test), "X_test and y_test are not equal in size."

    # Predict Test Data 
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average='macro')
    rec = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_prob = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    auc = metrics.roc_auc_score(y_test, y_pred_prob)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print('Kappa:', kappa)
    #print('Confusion Matrix:\n', knn_eval['confusion-matrix'])
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f'AUC: {round(auc, 2)}')
    plt.legend()
    plt.show()
    
    cm_norm = cm / cm.sum(axis=1).reshape(-1,1)
    plt.figure(figsize=(12,10))
    plot_confusion_matrix(cm_norm, title='Confusion matrix')
    plt.show()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'F1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'area': auc, 'confusion-matrix': cm}


# #Preprocessing

# In[ ]:


#Read file
df = pd.read_csv('heart_2020_cleaned.csv')

#Replace non-numeric answers
df =  df[df.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0,'No, borderline diabetes':'0','Yes (during pregnancy)':'1' })
df['Diabetic'] = df['Diabetic'].astype(int)
binary_cols = ['HeartDisease','Sex','Smoking','AlcoholDrinking','Stroke','Asthma', 'DiffWalking','PhysicalActivity','KidneyDisease','SkinCancer']

#Standardize
num_cols = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
Scaler = StandardScaler()
df[num_cols] = Scaler.fit_transform(df[num_cols])
df = pd.get_dummies(df, columns = ['AgeCategory', 'Race',  'Diabetic', 'GenHealth'])

#Seed value
Seed = 42

#Split data
x = df.drop(columns = ['HeartDisease'], axis = 1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle = True, test_size = 0.2, random_state = Seed)

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)

df


# In[ ]:





# # Cross Validation

# In[ ]:


#Implementing cross validation using k fold of 5
k = 5
kf = KFold(n_splits=k, random_state=None)
model = LogisticRegression(solver= 'liblinear')
 
acc_score = []
 
for train_index , test_index in kf.split(x):
    X_train , X_test = x.iloc[train_index,:],x.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))


# # Exploratory Analysis

# In[ ]:


#importing the data without modification for preliminary visualisations
ds = pd.read_csv('heart_2020_cleaned.csv')
ds_cat = ds.select_dtypes(include="object")

ds


# In[ ]:


#categorise data by age
ds1 = ds.sort_values(by = 'AgeCategory' , ascending = True)
ds1


# In[ ]:


# data visualization for gender and age
gender_counts = ds.Sex.value_counts()
gender_counts
plt.title('Gender Count')
sns.barplot(x=gender_counts.index , y = gender_counts);


plt.figure(figsize = (8,4))
plt.title('Distribution of Age and Sex')
plt.xticks(rotation = 25)
sns.countplot(x = 'AgeCategory' ,hue='Sex' , data = ds1 );


# In[ ]:


# data visualization for each non numberical [arameter with respect to heart diseases
for col in ds_cat.columns[1:]:
    fig, ax = plt.subplots()
    ds[col][ds_cat["HeartDisease"] == "Yes"].value_counts().plot.bar()
    plt.title(f"Frequency distribution of {col} rates\n"
              f"of people with heart disease")


# In[ ]:


#Frequency Distribution of people with heart diseases
plt.figure(figsize=(8, 4))
plt.title("Frequency distribution of people with heart diseases")
sns.countplot(x = 'HeartDisease', data = ds)


# # Correlation Feature Selection

# In[ ]:


# Finding the correlation between the parameters and heart diseases
cols = df.columns
corr = []
for col in cols:
    if col in binary_cols:
        corr.append(matthews_corrcoef(df['HeartDisease'], df[col]))
    else:
        corr.append(df['HeartDisease'].corr(df[col]))
correlation = pd.DataFrame(list(zip(cols, corr)), columns=['Variables','Corr_with_HeartDisease'])
correlation.drop(index=correlation.index[:1], axis=0, inplace=True)
correlation
    


# In[ ]:


# Plotting the correlation
for i in range(len(correlation)):
    # Colour of bar chart is set to red if corr is < 0 and green otherwise
    correlation['colors'] = ['#F0073B' if float(x) <= 0 else '#CB3579' for x in correlation['Corr_with_HeartDisease']]
correlation = correlation.sort_values(ascending=True, by=['Corr_with_HeartDisease']) 
plt.figure(figsize=(14,10), dpi=80)
plt.hlines(y=correlation.Variables, xmin=0, xmax=correlation.Corr_with_HeartDisease, color=correlation.colors,  linewidth=5)
plt.grid()
plt.show()


# #SVM

# In[ ]:


# Create Model using pipeline to combine preprocess with SVC and parameters 
model = make_pipeline(StandardScaler(), SVC(kernel = "rbf", gamma="scale", max_iter = 125, probability = True))
svm_fit = model.fit(X_train, y_train)

# Call evaluate function and display connfusion matrix
evaluate_model(svm_fit, X_test, y_test)


# Create SVM with Balanced Class Weighting
model = make_pipeline(StandardScaler(), SVC(kernel = "rbf", gamma="scale",class_weight = "balanced", max_iter = 500, probability = True))
svm_fit = model.fit(X_train, y_train)

# Unbalanced Results
evaluate_model(svm_fit, X_test, y_test)


# #KNN

# In[ ]:


# Running KNN for K values from K = 2 to 9
for k in range(2,9):
  print("Running KNN for K =", k)
  
  #Create model
  knn = KNeighborsClassifier(n_neighbors = k)
  
  #Fit Model
  knn.fit(X_train, y_train)
  
  #Evaluate Model
  knn_eval = evaluate_model(knn, X_test, y_test)


#  #  Logistic Regression

# Unbalanced Data

# In[ ]:


# Logistic Regression on unbalanced data
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)


# In[ ]:


# Calling the function to plot the results
evaluate_model(lr, X_test, y_test)


# In[ ]:


# printing the accuracy
print('The training accuracy is', lr.score(X_train, y_train))
print('The test accuracy is', lr.score(X_test, y_test))


# Balanced Data

# In[ ]:


# Logistic Regression on balanced data

lr_bal = LogisticRegression(max_iter=500, class_weight= "balanced")
lr_bal.fit(X_train, y_train)


# In[ ]:


evaluate_model(lr_bal, X_test, y_test)


# In[ ]:


print('The training accuracy is', lr_bal.score(X_train, y_train))
print('The test accuracy is', lr_bal.score(X_test, y_test))


# Undersampled

# In[ ]:


# Logistic Regression using undersampling

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy="majority")
X_under, y_under = undersample.fit_resample(x, y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, random_state=42)
lr_us = LogisticRegression(max_iter=500)
lr_us.fit(X_train, y_train)

print('The training accuracy is', lr_us.score(X_train, y_train))
print('The test accuracy is', lr_us.score(X_test, y_test))


# In[ ]:


evaluate_model(lr_us, X_test, y_test)


# Undersampled Balanced

# In[ ]:


# Logistic Regression on balanced data using undersampling

X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, random_state=42)
lr_us = LogisticRegression(max_iter=500, class_weight = "balanced")
lr_us.fit(X_train, y_train)

print('The training accuracy is', lr_us.score(X_train, y_train))
print('The test accuracy is', lr_us.score(X_test, y_test))


# In[ ]:


evaluate_model(lr_us, X_test, y_test)


# In[ ]:


print('The training accuracy is', lr_us.score(X_train, y_train))
print('The test accuracy is', lr_us.score(X_test, y_test))


# # Balancing the data initially

# In[ ]:



# Balancing the data initially
class_0 = df[df['HeartDisease'] == 0]
class_1 = df[df['HeartDisease'] == 1]

class_1 = class_1.sample(len(class_0),replace=True)
df = pd.concat([class_0, class_1], axis=0)
print('Data in Train:')
print(df['HeartDisease'].value_counts())


# In[ ]:


x = df.drop(columns = ['HeartDisease'], axis = 1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle = True, test_size = 0.2, random_state = Seed)


# In[ ]:


# Create LogisticRegression
lr_bal2 = LogisticRegression(max_iter=500, random_state=Seed)
lr_bal2.fit(X_train, y_train)
lr_y_predict = lr_bal2.predict(X_test)


# In[ ]:


evaluate_model(lr_bal2, X_test, y_test)


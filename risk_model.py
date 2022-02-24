# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:26:33 2022

@author: hua'wei

补 LR+NB


"""



# 决策树和随机森林
# 共1004例样本
# 训练集703（527:176）  
# 测试集301（226:75）
# 测试集和训练集的不均衡比均约为1:3



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,recall_score, roc_auc_score, f1_score, roc_curve,precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import imblearn   # 处理不均衡的库
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
import seaborn as sns #导入包：用seaborn的热力图画混淆矩阵图

# 加载数据
data_train = pd.read_csv('train_data703.csv')
data_test = pd.read_csv('test_data301.csv')
data_train.drop('Unnamed: 0',axis=1,inplace=True)    
data_test.drop('Unnamed: 0',axis=1,inplace=True)     
data_train.drop(['原始序号','心绞痛稳定状态','治疗满意程度','疾病认知程度','PHQ9','GAD7','SAS','SDS','抽烟_2','抽烟_0','PCI次数_0','PCI次数_1','PCI次数_2','PCI次数_3','PCI次数_4'],axis=1,inplace=True)     # 第1行是标识列，删除
data_test.drop(['原始序号','心绞痛稳定状态','治疗满意程度','疾病认知程度','PHQ9','GAD7','SAS','SDS','抽烟_2','抽烟_0','PCI次数_0','PCI次数_1','PCI次数_2','PCI次数_3','PCI次数_4'],axis=1,inplace=True)     # 第1行是标识列，删除

data_train_x = data_train.iloc[:,0:-1]    
data_train_y = data_train.iloc[:,-1]      
data_test_x = data_test.iloc[:,0:-1]   
data_test_y = data_test.iloc[:,-1]     

sm = SMOTE(random_state=42)
data_train_x,data_train_y = sm.fit_resample(data_train_x,data_train_y)
data_test_x,data_test_y = sm.fit_resample(data_test_x,data_test_y)

Xtrain = data_train_x
Ytrain = data_train_y
Xval = data_test_x
Yval = data_test_y

clf_DT = DecisionTreeClassifier(criterion='entropy'
                                  ,random_state=25     # 参数：random_state用来设置分枝中的随机模式，默认None
                                  ,splitter='random'  # 防止过拟合，当特征很多的时候，最好加上,效果不好的话注掉
                                  ,max_depth=9
                                  ,min_samples_leaf=8   #叶子节点中最少包含的样本数
                                  ,min_samples_split=8   #分枝节点最少包含的样本数
                                  #,class_weight='balanced'    # 自动给不均衡样本 加权处理
                                  #,min_weight_fraction_leaf = 0.001
                                  )
clf_RF = RandomForestClassifier(random_state=80
                             ,n_estimators=30
                             #,class_weight='balanced'    # 自动给不均衡样本 加权处理
                             ,max_depth=9
                             ,min_samples_leaf=8
                             ,min_samples_split=8
                             ,criterion='entropy'
                             )


clf_DT = clf_DT.fit(Xtrain,Ytrain)
clf_RF = clf_RF.fit(Xtrain,Ytrain)
train_accuracy_DT = clf_DT.score(Xtrain,Ytrain)
train_accuracy_RF = clf_RF.score(Xtrain,Ytrain)
val_accuracy_DT = clf_DT.score(Xval,Yval)
val_accuracy_RF = clf_RF.score(Xval,Yval)

# print("Single DecisionTree 训练精度: {}\n".format(train_accuracy_DT)
#       ,"Random Forest 训练精度: {}".format(train_accuracy_RF))
# print("Single DecisionTree 测试精度: {}\n".format(val_accuracy_DT)
#       ,"Random Forest 测试精度: {}".format(val_accuracy_RF))


pre_DT = clf_DT.predict(Xval)
pre_RF = clf_RF.predict(Xval)

CM_DT = confusion_matrix(Yval,pre_DT,labels=[0,1])
CM_RF = confusion_matrix(Yval,pre_RF,labels=[0,1])
print('Decision Tree 混淆矩阵：\n',CM_DT)
print('Random Forest 混淆矩阵：\n',CM_RF)

R_DT = recall_score(Yval,pre_DT)
R_RF = recall_score(Yval,pre_RF)
print('Decision Tree 召回率：\n',R_DT)
print('Random Forest 召回率：\n',R_RF)

F1_DT = f1_score(Yval,pre_DT)
F1_RF = f1_score(Yval,pre_RF)
print('Decision Tree F1：\n',F1_DT)
print('Random Forest F1：\n',F1_RF)

AUC_DT = roc_auc_score(Yval,clf_DT.predict_proba(Xval)[:,1])
AUC_RF = roc_auc_score(Yval,clf_RF.predict_proba(Xval)[:,1])
print('Decision Tree AUC值：\n',AUC_DT)
print('Random Forest AUC值：\n',AUC_RF)

feature_name = list(Xtrain)    # 所有的列名（特征名字）
feature_name = ['Work','TCM treatment','Family history','Seasonal onset','Bleeding events','Old Myocardial infarction',
                'Dyslipidemia','Brain infarction','Cardiac insufficiency','Anticoagulant drugs',
                'Antiarrhythmic drugs','Diuretic','Lansoprazole injection','Bypass surgery','Smoking','Age',
                'Course of the disease','HAMD','HAMA','LAD','LVEF']

f = clf_DT.feature_importances_
fn = (f-f.min())/(f.max()-f.min())
feature_importance_DT = [*zip(feature_name,fn)]
feature_importance_DT = sorted(feature_importance_DT, key=itemgetter(1), reverse=False)  # reverse=False降序
# print('--------特征重要性------\n',feature_importance_DT) 

value =[]
for i in feature_importance_DT:
    value.append(i[1])
    
name =[]
for i in feature_importance_DT:
    name.append(i[0])

plt.figure()
plt.barh(name,value)
plt.title('DT')
plt.show()


f = clf_RF.feature_importances_
fn = (f-f.min())/(f.max()-f.min())
feature_importance_RF = [*zip(feature_name,fn)]
feature_importance_RF = sorted(feature_importance_RF, key=itemgetter(1), reverse=False) 
# print('--------特征重要性------\n',feature_importance_RF) 

value =[]
for i in feature_importance_RF:
    value.append(i[1])
    
name =[]
for i in feature_importance_RF:
    name.append(i[0])

plt.figure()
plt.barh(name,value)
plt.title('RF')
plt.show()


clf_XGB = XGBClassifier(#scale_pos_weight=3
                              n_estimators=15
                              ,max_depth=6
                              ,splitter='random'
                              ,objective='binary:logistic'
                              ,eta=0.3
                            )
clf_XGB = clf_XGB.fit(Xtrain,Ytrain)
# pre_y = clf_xgb.predict(Xtest)
train_accuracy_XGB = clf_XGB.score(Xtrain,Ytrain)
val_accuracy_XGB = clf_XGB.score(Xval,Yval)
# print('xgboost训练精度：\n',train_accuracy_XGB)
# print('xgboost预测精度：\n',val_accuracy_XGB)
# 预测的y值
pre_XGB = clf_XGB.predict(Xval)

CM_XGB = confusion_matrix(Yval,pre_XGB,labels=[0,1])   # 少数类写在前面
# print('XGBoost 混淆矩阵：\n',CM_XGB)
R_XGB = recall_score(Yval,pre_XGB)
# print('XGBppst 召回率：\n',R_XGB)
F1_XGB = f1_score(Yval,pre_XGB)
# print('XGBoost F1：\n',F1_XGB)
AUC_XGB = roc_auc_score(Yval,clf_XGB.predict_proba(Xval)[:,1])
# print('XGBoost AUC值：\n',AUC_XGB)

f = clf_XGB.feature_importances_
fn = (f-f.min())/(f.max()-f.min())
feature_importance_XGB = [*zip(feature_name,fn)]
feature_importance_XGB = sorted(feature_importance_XGB, key=itemgetter(1), reverse=False) 
# print('--------特征重要性------\n',feature_importance_XGB) 


value =[]
for i in feature_importance_XGB:
    value.append(i[1])
    
name =[]
for i in feature_importance_XGB:
    name.append(i[0])

plt.figure()
plt.barh(name,value)
plt.title('XGBoost')
plt.show()


clf_LR = LogisticRegression(#class_weight='balanced'
                            )
clf_LR = clf_LR.fit(Xtrain,Ytrain)
pre_LR = clf_LR.predict(Xval)
train_accuracy_LR = clf_LR.score(Xtrain,Ytrain)
val_accuracy_LR = clf_LR.score(Xval,Yval)
# print('预测的y值：\n',pre_y)
# print('logistic regression 训练精度：\n',clf_LR.score(Xtrain,Ytrain))
# print('logistic Regression 预测精度：\n',val_accuracy_LR)

CM_LR = confusion_matrix(Yval,pre_LR,labels=[0,1])
# print('混淆矩阵：\n',CM_LR)
R_LR = recall_score(Yval,pre_LR)
# print('召回率：\n',R_LR)
AUC_LR = roc_auc_score(Yval,clf_LR.predict_proba(Xval)[:,1])
# print('AUC值：\n',AUC_LR)
F1_LR = f1_score(Yval,pre_LR)
# print('XGBoost F1：\n',F1_LR)

f = clf_LR.coef_.flatten()
fn = (f-f.min())/(f.max()-f.min())
feature_importance_LR = [*zip(feature_name,fn)]
feature_importance_LR = sorted(feature_importance_LR, key=itemgetter(1), reverse=False) 
# print('--------特征重要性------\n',feature_importance_XGB) 
value =[]
for i in feature_importance_LR:
    value.append(i[1])
    
name =[]
for i in feature_importance_LR:
    name.append(i[0])

plt.figure()
plt.barh(name,value)
plt.title('LR')
plt.show()

clf_SVC = SVC(kernel='rbf'
              # ,gamma='auto'
              # ,degree=1     # degree=1:多项式核函数的次数：线性；degree>1：多项式非线性；默认3
              # ,cache_size=1000     # 使用多大的内存进行计算，默认200MB
               ,probability = True
        )
clf_SVC = clf_SVC.fit(Xtrain,Ytrain)
print('**********************')
train_accuracy_SVC = clf_SVC.score(Xtrain,Ytrain)
val_accuracy_SVC = clf_SVC.score(Xval,Yval)   # 预测精度
# print('SVM 训练精度：\n',train_accuracy_SVC)
# print('SVM 预测精度:',val_accuracy_SVC)
pre_SVC = clf_SVC.predict(Xval) # 预测的y值
CM_SVC = confusion_matrix(Yval,pre_SVC,labels=[0,1])
# print('混淆矩阵：\n',CM_SVC)
R_SVC = recall_score(Yval,pre_SVC)
# print('召回率：\n',R_SVC)
F1_SVC = f1_score(Yval,pre_SVC)
# print('XGBoost F1：\n',F1_SVC)
AUC_SVC = roc_auc_score(Yval,clf_SVC.predict_proba(Xval)[:,1])
# print('AUC值：\n',AUC_SVC)

clf_NB = GaussianNB()
#clf_NB = BernoulliNB()
clf_NB = clf_NB.fit(Xtrain,Ytrain)
print('**********************')
train_accuracy_NB = clf_NB.score(Xtrain,Ytrain)
val_accuracy_NB = clf_NB.score(Xval,Yval)   # 预测精度
# print('NB 训练精度：\n',train_accuracy_NB)
# print('NB 预测精度:',val_accuracy_NB)
pre_NB = clf_NB.predict(Xval) # 预测的y值
CM_NB = confusion_matrix(Yval,pre_NB,labels=[0,1])
# print('混淆矩阵：\n',CM_NB)
R_NB = recall_score(Yval,pre_NB)
# print('召回率：\n',R_NB)
F1_NB = f1_score(Yval,pre_NB)
# print('XGBoost F1：\n',F1_NB)
AUC_NB = roc_auc_score(Yval,clf_NB.predict_proba(Xval)[:,1])
# print('AUC值：\n',AUC_NB)

#--------画图：模型对比-------
precision_DT = precision_score(Yval,pre_DT)
precision_RF = precision_score(Yval,pre_RF)
precision_LR = precision_score(Yval,pre_LR)
precision_NB = precision_score(Yval,pre_NB)
precision_SVC = precision_score(Yval,pre_SVC)
precision_XGB = precision_score(Yval,pre_XGB)

# 不同机器学习模型性能（柱形图）
x_name = ['Accuracy','Precision','Recall','F1','AUC']
x = np.arange(len(x_name))
DT = [val_accuracy_DT,precision_DT,R_DT,F1_DT,AUC_DT]
RF = [val_accuracy_RF,precision_RF,R_RF,F1_RF,AUC_RF]
LR = [val_accuracy_LR,precision_LR,R_LR,F1_LR,AUC_LR]
NB = [val_accuracy_NB,precision_NB,R_NB,F1_NB,AUC_NB]
SVM = [val_accuracy_SVC,precision_SVC,R_SVC,F1_SVC,AUC_SVC]
XGB = [val_accuracy_XGB,precision_XGB,R_XGB,F1_XGB,AUC_XGB]


total_width, n = 1.2,8
width = total_width/n
x = x - (total_width - width)/2

plt.figure()
plt.bar(x,DT,width,label='DT')
plt.bar(x+width,RF,width,label='RF')
plt.bar(x+2*width,LR,width,label='LR')
plt.bar(x + 3 * width, NB, width=width,label='NB')
plt.bar(x+4*width,SVM,width,label='SVM')
plt.bar(x + 5 * width, XGB, width=width,label='XGBoost')
plt.xticks(x+width+width+width/2,x_name)
plt.legend()
plt.show()
print('--------------------results----------------')
# print('\n\n Train_Accuracy:\n DT: {}\n RF: {} \n LR: {}\n NB: {}\n SVM: {} \n XGB: {}'.format(train_accuracy_DT,train_accuracy_RF,train_accuracy_LR,train_accuracy_NB,train_accuracy_SVC,train_accuracy_XGB))
print('\n\n Test_Accuracy:\n DT: {}\n RF: {} \n LR: {}\n NB: {}\n SVM: {} \n XGB: {}'.format(val_accuracy_DT,val_accuracy_RF,val_accuracy_LR,val_accuracy_NB,val_accuracy_SVC,val_accuracy_XGB))
print('\n\nPrecision:\n DT: {}\n RF: {} \n LR: {}\n NB: {}\nSVM: {} \n XGB: {}'.format(precision_DT,precision_RF,precision_LR,precision_NB,precision_SVC,precision_XGB))
print('\n\nRecall:\n DT: {}\n RF: {} \n LR: {}\n NB: {}\nSVM: {} \n XGB: {}'.format(R_DT,R_RF,R_LR,R_NB,R_SVC,R_XGB))
print('\n\nF1_score:\n DT: {}\n RF: {}\n LR: {}\n NB: {}\nSVM: {} \n XGB: {}'.format(F1_DT,F1_RF,F1_LR,F1_NB,F1_SVC,F1_XGB))
print('\n\nAUC:\n DT: {}\n RF: {} \n LR: {}\n NB: {}\nSVM: {} \n XGB: {}'.format(AUC_DT,AUC_RF,AUC_LR,AUC_NB,AUC_SVC,AUC_XGB))

# --------ROC 曲线
yp_DT = clf_DT.predict_proba(Xval)[:,1]
yp_RF = clf_RF.predict_proba(Xval)[:,1]
yp_LR = clf_LR.predict_proba(Xval)[:,1]
yp_NB = clf_NB.predict_proba(Xval)[:,1]
yp_SVC = clf_SVC.predict_proba(Xval)[:,1]
yp_XGB = clf_XGB.predict_proba(Xval)[:,1]
fpr_DT, tpr_DT, thresholds_DT = roc_curve(Yval, yp_DT)
fpr_RF, tpr_RF, thresholds_RF = roc_curve(Yval, yp_RF)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(Yval, yp_LR)
fpr_NB, tpr_NB, thresholds_NB = roc_curve(Yval, yp_NB)
fpr_SVC, tpr_SVC, thresholds_SVC = roc_curve(Yval, yp_SVC)
fpr_XGB, tpr_XGB, thresholds_XGB = roc_curve(Yval, yp_XGB)

plt.figure()
# plt.plot(fpr_DT, tpr_DT,label='DT=%0.2f' % AUC_DT)
plt.plot(fpr_DT, tpr_DT,label='DT')
plt.plot(fpr_RF, tpr_RF,label='RF')
plt.plot(fpr_LR, tpr_LR,label='LR')
plt.plot(fpr_NB, tpr_NB,label='NB')
plt.plot(fpr_SVC, tpr_SVC,label='SVM')
plt.plot(fpr_XGB, tpr_XGB,label='XGBoost')

# plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

##  ---------混淆矩阵----
print('混淆矩阵 DT：\n',CM_DT)
print('混淆矩阵 RF：\n',CM_RF)
print('混淆矩阵 LR：\n',CM_LR)
print('混淆矩阵 NB：\n',CM_NB)
print('混淆矩阵 SVM：\n',CM_SVC)
print('混淆矩阵 XGB：\n',CM_XGB)
print('\n\n混淆矩阵:\n DT:\n {}\n RF:\n {}\n LR:\n {}NB:\n {}\nSVM:\n {} \n XGB:\n {}'.format(CM_DT,CM_RF,CM_LR,CM_NB,CM_SVC,CM_XGB))
plt.figure()
sns.heatmap(pd.DataFrame(CM_DT),fmt='g',annot=True,cmap='Blues')   
plt.title('DT')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.figure()
sns.heatmap(pd.DataFrame(CM_RF),fmt='g',annot=True,cmap='Blues')    
plt.title('RF')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.figure()
sns.heatmap(pd.DataFrame(CM_LR),fmt='g',annot=True,cmap='Blues') 
plt.title('LR')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.figure()
sns.heatmap(pd.DataFrame(CM_NB),fmt='g',annot=True,cmap='Blues')   
plt.title('NB')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.figure()
sns.heatmap(pd.DataFrame(CM_SVC),fmt='g',annot=True,cmap='Blues')    
plt.title('SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.figure()
sns.heatmap(pd.DataFrame(CM_XGB),fmt='g',annot=True,cmap='Blues')    
plt.title('XGB')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')



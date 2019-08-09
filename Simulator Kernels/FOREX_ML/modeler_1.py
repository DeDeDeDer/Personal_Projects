from FOREX_ML import Models_1 as mod
from FOREX_ML import Feature_functions_1 as feateng

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import plotly as py
from plotly import tools
import plotly.graph_objs as go
import seaborn as sns
from matplotlib import pyplot as pyplt

import re
from celluloid import Camera

"""Misc"""
path_0 = '/Users/Derrick-Vlad-/Desktop/'

def changeMe(col):
    obj = pd.np.array(col).tolist()
    new = [i[0] for i in obj]
    return new


def data_roc_auc(link):
    df = pd.read_csv(link, decimal='.')
    import re
    models = ['LASSO', 'Ridge', 'Logistic Balance',	'Support Vector Machine', 'Random Forest Classifier', 'Gradient Boost Classifier',  'Naive Bayes',	'Gaussian Classifier', 'K Nearest Classifier']
    for m in range(0, len(models)):
        mod = models[m]

        df[mod] = df[mod].apply(lambda x: re.sub("\s+", ",", x.strip()).replace(']','').replace('[','').split(","))

    print('Completed')
    return df['LASSO'], df['Ridge'], df['Logistic Balance'], df['Support Vector Machine'], df['Random Forest Classifier'], df['Gradient Boost Classifier'], df['Naive Bayes'], df['Gaussian Classifier'], df['K Nearest Classifier']


def ls_arrays_str_float(data):
    ls = []
    for i in range(0, len(data)):
        filter_blanks = [x for x in data[i] if x != '']
        new_list = list(map(float, filter_blanks))
        arr = np.array(new_list)
        ls.append(arr)
    return ls





"""
/////////////////////////////////////////
Read & Clean Data
/////////////////////////////////////////
"""
# Read & Clean Data
FK = '/Users/Derrick-Vlad-/Downloads/EURSGD_Candlestick_1_Hour_BID_31.12.2009-08.06.2019.csv'
df = pd.read_csv(FK)
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df['date'] = pd.to_datetime([x[:-9] for x in df['date'].squeeze().tolist()], dayfirst=True)
df.date = pd.to_datetime(df.date, format='%d.%m.%Y %H:%M:%S.%f')
# Set New Index
df.index = range(len(df))
# Rename & Drop Duplicates
df = df[['open', 'high', 'low', 'close', 'volume']]
df = df.drop_duplicates(keep=False)
df_close = df[['close']]
# Limit dataset for testing
END_DATA_LIMIT = 2000#6000       5000      10000     20000
df_close = df_close.iloc[:END_DATA_LIMIT]
df = df.iloc[:END_DATA_LIMIT]


"""
CROSS SECTIONAL
/////////////////////////////////////////
Random Forest Regression       note:NOT Classifier
/////////////////////////////////////////
"""
# Exogenous: Feature 0
feat_0 = df.high

# Exogenous: Feature 1
c = feateng.Holder.cci(df, [30])
feat_1 = c.CCI[30]

# Exogenous: Feature 2
s = feateng.Holder.stochastic(df, [14])
feat_2 = s.close[14].K

# Exogenous: Feature 3
m = feateng.Holder.momentum(df, [10])
feat_3 = m.close[10].close


# Endogenous: Higher Highs
trend = mod.Rules.h_highs_generic(raw_data=df, col_name='high', periods=[4, 6, 10], inter=2)
ut = trend.hh[4]
prop = trend.hh[4].sum().values / END_DATA_LIMIT
print('Generic Count ', str(prop), '\n')


# Compile
df_new = pd.concat((ut, feat_0, feat_1, feat_2, feat_3), axis=1)
df_new.columns = [['Uptrend', 'close', 'CCI', 'Stochastic', 'Momentum']]
df_new.dropna(inplace=True)
# df_new.index = np.arange(len(df_new))



# Correlation Heat-Map - Only int/floats
# cor = df_new.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(cor, annot=True, cmap='YlGn', fmt='.2f')
# plt.show()





# Model ALL ENSEMBLE
F = mod.Models.mod_ensemble_class(y_endogenous=df_new['Uptrend'],
                                  x_exogenous=df_new[['CCI', 'Stochastic', 'Momentum']],
                                  train_ratio=0.7)


F.f1_score.to_csv(path_0+'F1_Scores.csv', encoding='utf-8')
F.logloss.to_csv(path_0+'Logloss_Scores.csv', encoding='utf-8')
F.acc_score.to_csv(path_0+'Accuracy_Scores.csv', encoding='utf-8')


"""REMOVE THIS"""
F2 = mod.Models.mod_knn_class(y_endogenous=df_new['Uptrend'],
                              x_exogenous=df_new[['CCI', 'Stochastic', 'Momentum']],
                              train_ratio=0.7)


F2.f1_score.to_csv(path_0+'F1_KNN.csv', encoding='utf-8')
F2.logloss.to_csv(path_0+'Logloss_KNN.csv', encoding='utf-8')
F2.acc_score.to_csv(path_0+'Accuracy_KNN.csv', encoding='utf-8')



newlk = '/Users/Derrick-Vlad-/Desktop/LinkedIn Article 2/20000/ALL/'
oldlk = '/Users/Derrick-Vlad-/Desktop/TPRS_ALL.csv'
lasso_tp, ridge_tp, log_tp, svm_tp, rf_tp, gb_tp, nb_tp, gc_tp, knn_tp = data_roc_auc(link=newlk+'TPRS.csv')
lasso_fp, ridge_fp, log_fp, svm_fp, rf_fp, gb_fp, nb_fp, gc_fp, knn_fp = data_roc_auc(link=newlk+'FPRS.csv')


model_names = ['LASSO', 'Ridge', 'Logistic Balanced',
               'Support Vector Machine', 'Random Forest Classifier', 'Gradient Boost Classifier',
               'Naive Bayes', 'Gaussian Classifier',
               'K Nearest Classifier']

lasso_TPS = ls_arrays_str_float(lasso_tp)
lasso_FPS = ls_arrays_str_float(lasso_fp)

ridge_TPS = ls_arrays_str_float(ridge_tp)
ridge_FPS = ls_arrays_str_float(ridge_fp)

log_TPS = ls_arrays_str_float(log_tp)
log_FPS = ls_arrays_str_float(log_fp)

svm_TPS = ls_arrays_str_float(svm_tp)
svm_FPS = ls_arrays_str_float(svm_fp)

rf_TPS = ls_arrays_str_float(rf_tp)
rf_FPS = ls_arrays_str_float(rf_fp)

gb_TPS = ls_arrays_str_float(gb_tp)
gb_FPS = ls_arrays_str_float(gb_fp)

nb_TPS = ls_arrays_str_float(nb_tp)
nb_FPS = ls_arrays_str_float(nb_fp)

gc_TPS = ls_arrays_str_float(gc_tp)
gc_FPS = ls_arrays_str_float(gc_fp)

knn_TPS = ls_arrays_str_float(knn_tp)
knn_FPS = ls_arrays_str_float(knn_fp)


tp = [lasso_TPS, ridge_TPS, log_TPS, svm_TPS, rf_TPS, gb_TPS, nb_TPS, gc_TPS, knn_TPS]
fp = [lasso_FPS, ridge_FPS, log_FPS, svm_FPS, rf_FPS, gb_FPS, nb_FPS, gc_FPS, knn_FPS]
for i in range(0, len(tp)):

    mod.ensemble_plot_all_roc_curve(fprs=fp[i], tprs=tp[i], model_name=model_names[i])







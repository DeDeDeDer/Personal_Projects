"""Structure"""
import numpy as np
import pandas as pd
"""Plot"""
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import plotly as py
from plotly import tools
import plotly.graph_objs as go
import matplotlib.pyplot as pyplt
import seaborn as sns
"""Data Cleaning"""
from datetime import datetime
"""Statistical & Models"""
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential, load_model
#from keras.layers import LSTM, Dense, Dropout
import os

from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy import interp


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import argrelextrema

"""Model Evaluation"""
from sklearn.metrics import (roc_curve, auc,
                             jaccard_similarity_score,
                             confusion_matrix,
                             f1_score, classification_report,
                             log_loss)


"""
###################################################
MODEL PARAMETERS
###################################################
"""


class Parameters:
    @staticmethod
    def plot_acf(raw_prices, n_lags=20):
        # Calculate natural logs
        p_Log = np.log(raw_prices)
        p_Log_diff = p_Log - p_Log.shift()
        p_Log_diff.dropna(inplace=True)
        # For Q-parameter
        lag_acf = acf(p_Log_diff, nlags=n_lags)
        # Plot
        pos_sd_ci = 1.96 / np.sqrt(len(p_Log_diff))
        neg_sd_ci = -1.96 / np.sqrt(len(p_Log_diff))
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=pos_sd_ci, linestyle='--', color='gray')
        plt.axhline(y=neg_sd_ci, linestyle='--', color='gray')
        plt.title('Autocorrelation Function')
        plt.show()


    @staticmethod
    def plot_pacf(raw_prices, n_lags=20):
        # Calculate natural logs
        px_Log = np.log(raw_prices)
        # Calculate natural logs Difference
        px_Log_diff = px_Log - px_Log.shift()
        px_Log_diff.dropna(inplace=True)
        # For P-parameter
        lag_pacf = pacf(px_Log_diff, nlags=n_lags)
        # Plot
        pos_sd_ci = 1.96 / np.sqrt(len(px_Log_diff))
        neg_sd_ci = -1.96 / np.sqrt(len(px_Log_diff))
        plt.subplot(121)
        plt.plot(lag_pacf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=pos_sd_ci, linestyle='--', color='gray')
        plt.axhline(y=neg_sd_ci, linestyle='--', color='gray')
        plt.title('Partial-autocorrelation Function')
        plt.show()


"""
###################################################
MODELS
###################################################
"""


class Models:
# Ensemble
    @staticmethod
    def mod_ensemble_class(y_endogenous, x_exogenous, train_ratio=0.7, folds=5):
        """
        :param y_endogenous:
        :param x_exogenous:
        :param train_ratio:
        :param folds:
        :return:
        """
        random_state = 123
        """Drop NaN"""
        y_endogenous.dropna(inplace=True)
        x_exogenous.dropna(inplace=True)
        """Transform data for LogReg fitting"""
        scaler = StandardScaler()
        std_data = scaler.fit_transform(x_exogenous.values)
        std_data = pd.DataFrame(std_data, index=x_exogenous.index, columns=x_exogenous.columns)
        """Shuffle Data for IMBALANCES"""
        from sklearn.utils import shuffle
        X_shuf, Y_shuf = shuffle(std_data, y_endogenous)
        X_shuf = X_shuf.as_matrix().astype(np.float)
        Y_shuf = Y_shuf.as_matrix().astype(np.int)
        """K-fold CV"""
        cv = StratifiedKFold(n_splits=folds, shuffle=False)

        """Establish Models Settings"""
        # White-Box: GLM
        lasso = LogisticRegression(penalty='l1', C=0.1, random_state=random_state, solver='liblinear', n_jobs=1)
        ridge = LogisticRegression(penalty='l2', C=0.1, random_state=random_state, solver='liblinear', n_jobs=1)
        log = LogisticRegression(class_weight='balanced', C=0.1, random_state=random_state, solver='liblinear', n_jobs=1)
        svc = SVC(C=0.1, kernel='linear', cache_size=100, shrinking=True, decision_function_shape='ovo', probability=True)
        # Black-Box: Bagging
        rfc = RandomForestClassifier(random_state=random_state,
                                     bootstrap=True, max_depth=80,
                                     criterion='entropy',
                                     min_samples_leaf=3, min_samples_split=10,
                                     n_estimators=500,
                                     max_features=None)
        gbc = GradientBoostingClassifier(learning_rate=0.5,
                                         n_estimators=250,
                                         min_samples_split=200,
                                         max_depth=3)
        # Non-Linear
        nb = GaussianNB()
        gpc = GaussianProcessClassifier()
        mnb = MultinomialNB()
        bnb = BernoulliNB(binarize=True)
        knn = KNeighborsClassifier(n_neighbors=2)

        """Storage List Dictionary for Models"""
        en_models = [

            {
                'label': 'LASSO',
                'model': lasso,
                'dict_metrics': {},
            },

            {
                'label': 'Ridge',
                'model': ridge,
                'dict_metrics': {},
            },

            {
                'label': 'Logistic Balance',
                'model': log,
                'dict_metrics': {},
            },

            {
                'label': 'Support Vector Machine',
                'model': svc,
                'dict_metrics': {},
            },

            {
                'label': 'Random Forest Classifier',
                'model': rfc,
                'dict_metrics': {},
            },

            {
                'label': 'Gradient Boost Classifier',
                'model': gbc,
                'dict_metrics': {},
            },

            {
                'label': 'Naive Bayes',
                'model': nb,
                'dict_metrics': {},
            },

            {
                'label': 'Gaussian Classifier',
                'model': gpc,
                'dict_metrics': {},
            },


            #
            # {
            #     'label': 'Multi Naive Bayes',
            #     'model': mnb,
            #     'dict_metrics': {},
            # },
            #
            # {
            #     'label': 'Bernoulli Naive Bayes',
            #     'model': bnb,
            #     'dict_metrics': {},
            # },


            # {
            #     'label': 'K Neighbors Classifier',
            #     'model': knn,
            #     'dict_metrics': {},
            # }

        ]

        """Loop Models"""
        for m in en_models:
            MOD = m['model']
            print(m['label'])
            # AUC storage
            mean_tprs_y, mean_fpr_x = [], np.linspace(0, 1, 100)
            fprs_x, tprs_y, aucs = [], [], []
            # Other Metrics Storage: Evaluation Metrics Dictionary
            dict_metrics = {
                'fold_no': [],      # 1
                'acc_score': [],    # 2
                'jaccard_ind': [],  # 3
                'conf_matrix': [],  # 4
                'f1_score': [],     # 5
                'log_loss': [],     # 6
                'feat_coef': [],    # 7
                'feat_names': [],   # 8
                'fprs': [],
                'tprs': []
            }
            # Train / Test Split
            i = 1   # Start Loop
            for train_ind, test_ind in cv.split(X_shuf, Y_shuf):

                """FOR THE ... ISSUE!!!!!!!"""
                np.set_printoptions(threshold=np.inf)


                # Train Test Split
                X_train, X_test = X_shuf[train_ind], X_shuf[test_ind]
                y_train, y_test = Y_shuf[train_ind], Y_shuf[test_ind]
                # Fit Model
                MOD.fit(X_train, y_train)
                # ROC Curve
                fpr, tpr, thresholds = roc_curve(y_test, MOD.predict_proba(X_test).T[1])
                roc_auc = auc(fpr, tpr)
                fprs_x.append(fpr)
                tprs_y.append(tpr)
                mean_tprs_y.append(interp(mean_fpr_x, fpr, tpr))
                aucs.append(roc_auc)

                # Fold Number
                fold_no = i

                # Accuracy Score
                y_pred = MOD.predict(X_test)
                acc_score = metrics.accuracy_score(y_test, y_pred)
                # Jaccard Index
                j_index = jaccard_similarity_score(y_true=y_test, y_pred=y_pred)
                j_index_rnd = round(j_index, 2)
                # Confusion Matricsw
                cm = confusion_matrix(y_test, y_pred)
                # F1 Score
                f1 = f1_score(y_test, y_pred)
                # Log Loss
                lg_loss = log_loss(y_test, y_pred)
                # Feature Importance
                try:
                    if m['label'] == 'Random Forest Classifier':
                        feature_imp = pd.Series(rfc.feature_importances_, index=x_exogenous.columns).sort_values(ascending=False)
                        feature_coef = pd.Series(feature_imp, index=x_exogenous.columns).sort_values(ascending=False)
                        dict_metrics['feat_coef'].append(feature_coef.values)
                        dict_metrics['feat_names'].append(feature_coef.index)
                    elif m['label'] == 'Gradient Boost Classifier':
                        feature_imp = pd.Series(gbc.feature_importances_, index=x_exogenous.columns).sort_values(ascending=False)
                        feature_coef = pd.Series(feature_imp, index=x_exogenous.columns).sort_values(ascending=False)
                        dict_metrics['feat_coef'].append(feature_coef.values)
                        dict_metrics['feat_names'].append(feature_coef.index)
                    elif m['label'] == 'none':
                        pass
                    else:
                        # Feature Coefficients
                        coefficients = MOD.coef_[0]
                        feature_coef = pd.Series(coefficients, index=x_exogenous.columns).sort_values(ascending=False)
                        dict_metrics['feat_coef'].append(feature_coef.values)
                        dict_metrics['feat_names'].append(feature_coef.index)
                except Exception:# (Valueerror, Attribute Error)
                    pass

                # Store Metrics
                dict_metrics['fold_no'].append(fold_no)
                dict_metrics['acc_score'].append(acc_score)
                dict_metrics['jaccard_ind'].append(j_index_rnd)
                dict_metrics['conf_matrix'].append(cm)
                dict_metrics['f1_score'].append(f1)
                dict_metrics['log_loss'].append(lg_loss)
                dict_metrics['fprs'].append(fpr)
                dict_metrics['tprs'].append(tpr)

                # Next Loop Indexer
                i = i + 1

            # Store All Metrics
            m['dict_metrics'] = dict_metrics

        """End??????"""
        labels = [i['label'] for i in en_models if 'label' in i]
        eva_all = [i['dict_metrics'] for i in en_models if 'dict_metrics' in i]
        accuracy = [i['acc_score'] for i in eva_all if 'acc_score' in i]
        f1 = [i['f1_score'] for i in eva_all if 'f1_score' in i]
        fprss = [i['fprs'] for i in eva_all if 'fprs' in i]
        tprss = [i['tprs'] for i in eva_all if 'tprs' in i]
        logL = [i['log_loss'] for i in eva_all if 'log_loss' in i]
        confmatrix = [i['conf_matrix'] for i in eva_all if 'conf_matrix' in i]
        # Prepare Data-frame
        # ACCURACY
        acc = np.vstack(accuracy)
        acc = np.transpose(acc)
        df1 = pd.DataFrame(acc, columns=labels)

        # F1 Score
        f1 = np.vstack(f1)
        f1 = np.transpose(f1)
        df2 = pd.DataFrame(f1, columns=labels)

        # FALSE POSITIVE RATES
        fprs = np.vstack(fprss)  # [:, 0] OR [:, None]
        fprs = np.transpose(fprs)
        df3 = pd.DataFrame(fprs, columns=labels)

        # TRUE POSITIVE RATES
        tprs = np.vstack(tprss)
        tprs = np.transpose(tprs)
        df4 = pd.DataFrame(tprs, columns=labels)

        # LOG LOSS SCORE
        logloss = np.vstack(logL)
        logloss = np.transpose(logloss)
        df5 = pd.DataFrame(logloss, columns=labels)

        # CONFUSION MATRIX
        #confmat = np.vstack(confmatrix)
        #confmat = np.transpose(confmat)
        #df6 = pd.DataFrame(confmat, columns=labels)
        print(confmatrix)

        results = Models()
        results.acc_score = df1
        results.f1_score = df2
        results.fprs = df3
        results.tprs = df4
        results.logloss = df5
        #results.confmat = df6

        return results


    @staticmethod
    def mod_knn_class(y_endogenous, x_exogenous, train_ratio=0.7, folds=5):
        """
        :param y_endogenous:
        :param x_exogenous:
        :param train_ratio:
        :param folds:
        :return:
        """
        random_state = 123
        """Drop NaN"""
        y_endogenous.dropna(inplace=True)
        x_exogenous.dropna(inplace=True)
        """Transform data for LogReg fitting"""
        scaler = StandardScaler()
        std_data = scaler.fit_transform(x_exogenous.values)
        std_data = pd.DataFrame(std_data, index=x_exogenous.index, columns=x_exogenous.columns)
        """Shuffle Data for IMBALANCES"""
        from sklearn.utils import shuffle
        X_shuf, Y_shuf = shuffle(std_data, y_endogenous)
        X_shuf = X_shuf.as_matrix().astype(np.float)
        Y_shuf = Y_shuf.as_matrix().astype(np.int)
        """K-fold CV"""
        cv = StratifiedKFold(n_splits=folds, shuffle=False)

        """Establish Models Settings"""
        # White-Box: GLM
        lasso = LogisticRegression(penalty='l1', C=0.1, random_state=random_state, solver='liblinear', n_jobs=1)
        ridge = LogisticRegression(penalty='l2', C=0.1, random_state=random_state, solver='liblinear', n_jobs=1)
        log = LogisticRegression(class_weight='balanced', C=0.1, random_state=random_state, solver='liblinear', n_jobs=1)
        svc = SVC(C=0.1, kernel='linear', cache_size=100, shrinking=True, decision_function_shape='ovo', probability=True)
        # Black-Box: Bagging
        rfc = RandomForestClassifier(random_state=random_state,
                                     bootstrap=True, max_depth=80,
                                     criterion='entropy',
                                     min_samples_leaf=3, min_samples_split=10,
                                     n_estimators=500,
                                     max_features=None)
        gbc = GradientBoostingClassifier(learning_rate=0.5,
                                         n_estimators=250,
                                         min_samples_split=200,
                                         max_depth=3)
        # Non-Linear
        nb = GaussianNB()
        gpc = GaussianProcessClassifier()
        mnb = MultinomialNB()
        bnb = BernoulliNB(binarize=True)
        knn = KNeighborsClassifier(n_neighbors=2)

        """Storage List Dictionary for Models"""
        en_models = [

            {
                'label': 'K Neighbors Classifier',
                'model': knn,
                'dict_metrics': {},
            }

        ]

        """Loop Models"""
        for m in en_models:
            MOD = m['model']
            print(m['label'])
            # AUC storage
            mean_tprs_y, mean_fpr_x = [], np.linspace(0, 1, 100)
            fprs_x, tprs_y, aucs = [], [], []
            # Other Metrics Storage: Evaluation Metrics Dictionary
            dict_metrics = {
                'fold_no': [],  # 1
                'acc_score': [],  # 2
                'jaccard_ind': [],  # 3
                'conf_matrix': [],  # 4
                'f1_score': [],  # 5
                'log_loss': [],  # 6
                'feat_coef': [],  # 7
                'feat_names': [],  # 8
                'fprs': [],
                'tprs': []
            }
            # Train / Test Split
            i = 1  # Start Loop
            for train_ind, test_ind in cv.split(X_shuf, Y_shuf):
                # Train Test Split
                X_train, X_test = X_shuf[train_ind], X_shuf[test_ind]
                y_train, y_test = Y_shuf[train_ind], Y_shuf[test_ind]
                # Fit Model
                MOD.fit(X_train, y_train)
                # ROC Curve
                fpr, tpr, thresholds = roc_curve(y_test, MOD.predict_proba(X_test).T[1])
                roc_auc = auc(fpr, tpr)
                fprs_x.append(fpr)
                tprs_y.append(tpr)
                mean_tprs_y.append(interp(mean_fpr_x, fpr, tpr))
                aucs.append(roc_auc)

                # Fold Number
                fold_no = i

                # Accuracy Score
                y_pred = MOD.predict(X_test)
                acc_score = metrics.accuracy_score(y_test, y_pred)
                # Jaccard Index
                j_index = jaccard_similarity_score(y_true=y_test, y_pred=y_pred)
                j_index_rnd = round(j_index, 2)
                # Confusion Matricsw
                cm = confusion_matrix(y_test, y_pred)
                # F1 Score
                f1 = f1_score(y_test, y_pred)
                # Log Loss
                lg_loss = log_loss(y_test, y_pred)
                # Feature Importance
                try:
                    if m['label'] == 'Random Forest Classifier':
                        feature_imp = pd.Series(rfc.feature_importances_, index=x_exogenous.columns).sort_values(
                            ascending=False)
                        feature_coef = pd.Series(feature_imp, index=x_exogenous.columns).sort_values(ascending=False)
                        dict_metrics['feat_coef'].append(feature_coef.values)
                        dict_metrics['feat_names'].append(feature_coef.index)
                    elif m['label'] == 'Gradient Boost Classifier':
                        feature_imp = pd.Series(gbc.feature_importances_, index=x_exogenous.columns).sort_values(
                            ascending=False)
                        feature_coef = pd.Series(feature_imp, index=x_exogenous.columns).sort_values(ascending=False)
                        dict_metrics['feat_coef'].append(feature_coef.values)
                        dict_metrics['feat_names'].append(feature_coef.index)
                    elif m['label'] == 'none':
                        pass
                    else:
                        # Feature Coefficients
                        coefficients = MOD.coef_[0]
                        feature_coef = pd.Series(coefficients, index=x_exogenous.columns).sort_values(ascending=False)
                        dict_metrics['feat_coef'].append(feature_coef.values)
                        dict_metrics['feat_names'].append(feature_coef.index)
                except Exception:  # (Valueerror, Attribute Error)
                    pass

                # Store Metrics
                dict_metrics['fold_no'].append(fold_no)
                dict_metrics['acc_score'].append(acc_score)
                dict_metrics['jaccard_ind'].append(j_index_rnd)
                dict_metrics['conf_matrix'].append(cm)
                dict_metrics['f1_score'].append(f1)
                dict_metrics['log_loss'].append(lg_loss)
                dict_metrics['fprs'].append(fpr)
                dict_metrics['tprs'].append(tpr)


                np.savetxt('/Users/Derrick-Vlad-/Desktop/' + 'FPR_KNN.csv', fpr, delimiter=",")
                np.savetxt('/Users/Derrick-Vlad-/Desktop/' + 'TPR_KNN.csv', tpr, delimiter=",")

                # Next Loop Indexer
                i = i + 1

            # Store All Metrics
            m['dict_metrics'] = dict_metrics

        """End??????"""
        labels = [i['label'] for i in en_models if 'label' in i]
        eva_all = [i['dict_metrics'] for i in en_models if 'dict_metrics' in i]
        accuracy = [i['acc_score'] for i in eva_all if 'acc_score' in i]
        f1 = [i['f1_score'] for i in eva_all if 'f1_score' in i]
        fprss = [i['fprs'] for i in eva_all if 'fprs' in i]
        tprss = [i['tprs'] for i in eva_all if 'tprs' in i]
        logL = [i['log_loss'] for i in eva_all if 'log_loss' in i]
        confmatrix = [i['conf_matrix'] for i in eva_all if 'conf_matrix' in i]
        # Prepare Data-frame
        # ACCURACY
        acc = np.vstack(accuracy)
        acc = np.transpose(acc)
        df1 = pd.DataFrame(acc, columns=labels)

        # F1 Score
        f1 = np.vstack(f1)
        f1 = np.transpose(f1)
        df2 = pd.DataFrame(f1, columns=labels)

        # FALSE POSITIVE RATES
        #fprs = np.vstack(fprss)  # [:, 0] OR [:, None]
        #fprs = np.transpose(fprs)
        #df3 = pd.DataFrame(fprs, columns=labels)
        print(fprss)

        # TRUE POSITIVE RATES
        #tprs = np.vstack(tprss)
        #tprs = np.transpose(tprs)
        #df4 = pd.DataFrame(tprs, columns=labels)
        print(tprss)

        # LOG LOSS SCORE
        logloss = np.vstack(logL)
        logloss = np.transpose(logloss)
        df5 = pd.DataFrame(logloss, columns=labels)

        # CONFUSION MATRIX
        # confmat = np.vstack(confmatrix)
        # confmat = np.transpose(confmat)
        # df6 = pd.DataFrame(confmat, columns=labels)
        print(confmatrix)

        results = Models()
        results.acc_score = df1
        results.f1_score = df2
        #results.fprs = df3
        #results.tprs = df4
        results.logloss = df5
        # results.confmat = df6

        return results
"""
###################################################
ENDOGENOUS PREDICTOR
###################################################
"""


class Rules:
# Rule 1 : Higher Highs
    @staticmethod
    def h_highs_generic(raw_data, col_name, periods, inter):
        """
        :param raw_data:
        :param col_name:
        :param periods:
        :param inter:
        :return:
        """
        df_high = raw_data[col_name]

        results = Rules()
        hh = {}

        for i in range(0, len(periods)):
            ms = []
            for j in range(periods[i], len(raw_data) - periods[i]):
                pxs = df_high.iloc[j - periods[i]:j:inter]
                # Either # np.all(np.diff(pxs) > 0)
                if all(i < j for i, j in zip(pxs, pxs[1:])):
                    m = 1
                else:
                    m = 0
                ms = np.append(ms, m)

            ms = pd.DataFrame(ms, index=df_high.iloc[periods[i] + 1:-periods[i] + 1].index)
            ms.columns = [['h_boo']]
            hh[periods[i]] = ms

        results.hh = hh
        return results


"""
###################################################
Misc
###################################################
"""


def ensemble_plot_all_roc_curve(fprs, tprs, model_name):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""

    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(7, 7))    # (14, 10)

    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.5,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random Case', alpha=.8)

    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='green',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.9)

    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.3,
                    label=r'$\pm$ 1 std. dev.')

    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic: ' + r"$\bf{" + model_name + "}$")
    ax.legend(loc="lower right")
    plt.show()
    return f, ax


def plot_all_feat_coef(dict, coefficients, names):
    # Prepare Data-frame
    conc_coef = np.vstack(dict[coefficients])
    conc_names = np.vstack(dict[names])
    df_coef = pd.DataFrame(conc_coef, columns=conc_names[0])
    df_coef['fold'] = range(1, len(df_coef)+1)

    # PIVOT Data-frame
    df_pivot = pd.melt(df_coef, id_vars=['fold'])

    # Plot
    ax = sns.factorplot(x="variable", y="value", hue='fold', data=df_pivot)
    ax.set(xlabel='Features', ylabel='Feature Importance Score', title='Visualizing Important Features')
    plt.show()


def plot_all_misc(dict, values, measure):
    # Prepare Data-frame
    conc_values = np.vstack(dict[values])
    df = pd.DataFrame(conc_values, columns=[measure])
    df['fold'] = range(1, len(df) + 1)
    print(df)
    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(df['fold'], df[measure])
    ax.set(xlabel='k-Fold', ylabel='Score', title='k-Fold ' + measure)
    limits = ['acc_score', 'f1_score']
    if any(values in s for s in limits):
        ax.set_xlim([-0.05, 1.05])
    else:
        pass
    plt.show()



def show_most_informative_features(vectorizer, clf, n):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))



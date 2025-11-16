import os
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# 导入kmeans
from sklearn.cluster import KMeans

# 导入PCA
from sklearn.decomposition import PCA

import imblearn
# 导入过采样
# 导入SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
# 导入欠采样
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

import xgboost
from xgboost import XGBClassifier

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

# 创建 pngs 目录（如果不存在）
os.makedirs('pngs', exist_ok=True)


def main():
    # 导入数据
    # https://archive-beta.ics.uci.edu/dataset/267/banknote+authentication
    banknote = pd.read_csv('./data/data_banknote_authentication.txt', header=None)
    banknote.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    print(banknote.head())

    # 可视化
    sns.pairplot(banknote, hue='class', size=2.5)
    plt.savefig('pngs/pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 数据探索
    print(banknote.describe())

    # 数据集中的数据都是数值型的，没有缺失值，也没有异常值，所以不需要进行数据清洗。
    # 但是我们可以看到，数据集中的数据都是数值型的，所以我们需要对数据进行标准化处理。
    # 为了方便后面的处理，我们将数据集分为特征集和标签集。
    X = banknote.drop('class', axis=1)
    y = banknote['class']

    # 使用不同的数据处理方法（如数据重采样、数据加权、特征选择等）来处理不均衡数据

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 检查数据集形状
    print('X_train shape: ', X_train.shape)
    print('X_test shape: ', X_test.shape)
    print('y_train shape: ', y_train.shape)
    print('y_test shape: ', y_test.shape)

    # 检查数据集标签
    print('y_train value counts: ', Counter(y_train))
    print('y_test value counts: ', Counter(y_test))

    # 不均衡数据对传统分类器的影响

    # 特征归一化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGboost训练
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print('XGboost准确率：', accuracy_score(y_test, y_pred_xgb))

    # KFold交叉验证评估XGboost模型
    kfold = KFold(n_splits=10, random_state=None)
    results_xgb = cross_val_score(xgb, X_train, y_train, cv=kfold)
    print('XGboost交叉验证准确率：', results_xgb.mean())

    # svm训练
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print('svm准确率：', accuracy_score(y_test, y_pred_svm))

    # KFold交叉验证评估svm模型
    kfold = KFold(n_splits=10, random_state=None)
    results_svm = cross_val_score(svm_model, X_train, y_train, cv=kfold)
    print('svm交叉验证准确率：', results_svm.mean())

    # RandomForest训练
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print('RandomForest准确率：', accuracy_score(y_test, y_pred_rf.astype(np.int64)))
    # predict和label的数据类型不同，predict还是float类型，表示分类类别的概率。label是int类型，代表的是类别标签

    # KFold交叉验证评估random forest模型
    kfold = KFold(n_splits=10, random_state=None)
    results_rf = cross_val_score(rf, X_train, y_train, cv=kfold)
    print('RandomForest交叉验证准确率：', results_rf.mean())

    # GradientBoosting训练
    gb = GradientBoostingRegressor()
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    print('GradientBoosting准确率：', accuracy_score(y_test, y_pred_gb.astype(np.int64)))
    # kfold交叉验证评估GradientBoosting模型
    kfold = KFold(n_splits=10, random_state=None)
    results_gb = cross_val_score(gb, X_train, y_train, cv=kfold)
    print('GradientBoosting交叉验证准确率：', results_gb.mean())

    # ExtraTreesRegressor训练
    et = ExtraTreesRegressor()
    et.fit(X_train, y_train)
    y_pred_et = et.predict(X_test)
    print('ExtraTreesRegressor准确率：', accuracy_score(y_test, y_pred_et.astype(np.int64)))

    # Kfold交叉验证评估ExtraTreesRegressor模型
    kfold = KFold(n_splits=10, random_state=None)
    results_et = cross_val_score(et, X_train, y_train, cv=kfold)
    print('ExtraTreesRegressor交叉验证准确率：', results_et.mean())

    # LinearRegression训练
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print('LinearRegression准确率：', accuracy_score(y_test, y_pred_lr.astype(np.int64)))

    # kfold交叉验证评估LinearRegression模型
    kfold = KFold(n_splits=10, random_state=None)
    results_lr = cross_val_score(lr, X_train, y_train, cv=kfold)
    print('LinearRegression交叉验证准确率：', results_lr.mean())

    # Ridge训练
    ridge = Ridge()
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    print('Ridge准确率：', accuracy_score(y_test, y_pred_ridge.astype(np.int64)))

    # kfold验证评估Ridge模型
    kfold = KFold(n_splits=10, random_state=None)
    results_ridge = cross_val_score(ridge, X_train, y_train, cv=kfold)
    print('Ridge交叉验证准确率：', results_ridge.mean())

    # Lasso训练
    lasso = Lasso()
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    print('Lasso准确率：', accuracy_score(y_test, y_pred_lasso.astype(np.int64)))

    # kfold交叉验证评估Lasso模型
    kfold = KFold(n_splits=10, random_state=None)
    results_lasso = cross_val_score(lasso, X_train, y_train, cv=kfold)
    print('Lasso交叉验证准确率：', results_lasso.mean())

    # SGDRegressor训练
    sgd = SGDRegressor()
    sgd.fit(X_train, y_train)
    y_pred_sgd = sgd.predict(X_test)
    print('SGDRegressor准确率：', accuracy_score(y_test, y_pred_sgd.astype(np.int64)))

    # kfold交叉验证评估SGDRegressor模型
    kfold = KFold(n_splits=10, random_state=None)
    results_sgd = cross_val_score(sgd, X_train, y_train, cv=kfold)
    print('SGDRegressor交叉验证准确率：', results_sgd.mean())

    # KernelRidge训练
    kr = KernelRidge()
    kr.fit(X_train, y_train)
    y_pred_kr = kr.predict(X_test)
    print('KernelRidge准确率：', accuracy_score(y_test, y_pred_kr.astype(np.int64)))

    # kfold交叉验证评估KernelRidge模型
    kfold = KFold(n_splits=10, random_state=None)
    results_kr = cross_val_score(kr, X_train, y_train, cv=kfold)
    print('KernelRidge交叉验证准确率：', results_kr.mean())

    # 对于未经处理的数据,我训练了包括KernelRidge Ridge SGDRegressor Lasso LinearRegression
    #  ExtraTreesRegressor GradientBoostingRegressor RandomForestRegressor SVM XGBoost在内的10个模型。

    # 我使用了交叉验证来评估模型的准确性。
    # 模型的得分如下：
    # SGDRegressor交叉验证准确率： 0.8631595873737876
    # KernelRidge交叉验证准确率： 0.043698643395488415
    # Lasso交叉验证准确率： -0.007906689939552302
    # Ridge交叉验证准确率： 0.8656096567425976
    # LinearRegression交叉验证准确率： 0.8656206880744337
    # ExtraTreesRegressor交叉验证准确率： 0.9908990575904794
    # GradientBoosting交叉验证准确率： 0.9554416627272119
    # RandomForest交叉验证准确率： 0.9616111476657251
    # svm交叉验证准确率： 1.0
    # XGboost交叉验证准确率： 0.9927083333333334

    # 接下来进行数据处理以应对数据集的不均衡性，这里使用了SMOTE算法，即合成少数类样本，使得数据集的样本数目达到均衡。
    # SMOTE算法的原理是，对于少数类样本中的每一个样本，都会找到与其最近邻的k个样本，然后从这k个样本中随机选择一个作为合成样本的特征。
    # 通过这种方法，可以使得少数类样本的数量达到均衡。
    # SMOTE算法的缺点是，合成的样本可能会带来噪声，因此在使用SMOTE算法之前，需要先对数据进行降维处理，这里使用了PCA算法。
    # PCA算法的原理是，对于一个n维的数据集，PCA算法会将其降维到k维，其中k<n，这样就可以减少数据集中的噪声。
    # 降维之后，再使用SMOTE算法，使得数据集的样本数目达到均衡。

    # PCA降维
    pca = PCA(n_components=0.9, whiten=True)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # SMOTE算法
    sm = SMOTE(random_state=42)
    # 调用SMOTE类中的fit_resample方法重新采样数据集
    X_train_smote, y_train_smote = sm.fit_resample(X_train_pca, y_train)

    # 数据集形状
    print('X_train_smote.shape:', X_train_smote.shape)
    print('y_train_smote.shape:', y_train_smote.shape)

    # 数据集标签检查
    print('y_train_smote:', Counter(y_train_smote))

    # 重新采样后的数据集标签分布
    plt.figure(figsize=(8, 6))
    sns.countplot(y_train_smote)
    plt.title('SMOTE后的数据集标签分布')
    plt.savefig('pngs/smote_label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("所有图片已保存到 pngs 目录")


if __name__ == "__main__":
    main()

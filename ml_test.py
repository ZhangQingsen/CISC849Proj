from sklearn import *
import time
from warnings import filterwarnings
filterwarnings('ignore')  # ignore warnings
feature_num = 16
class_num = 7

def knn_test(X,y):
    model_name = 'knn'
    time_start=time.time()

    k_range = range(1, 31)
    k_max = 0
    accu_max = 0
    print('start training KNN')
    #循环，取k=1到k=30，查看误差效果
    for k in k_range:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = model_selection.cross_validate(knn, X, y, scoring='accuracy')
        accu = scores['test_score'].mean()
        if accu >= accu_max:
            print('...')
            k_max = k
            accu_max = accu
    
    
    time_elapse = time.time()-time_start
    
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')
    print(f'the best accuracy k-value is: {k_max}, where the accuracy is: {accu_max}')
    
    return accu, time_elapse, model_name

def svm_test(X,y):
    model_name = 'svm'
    time_start=time.time()

    k_range = range(1, 31)
    gamma_max = 0
    accu_max = 0
    print('start training SVM')
    #循环，取k=1到k=30，查看误差效果
    for k in k_range:
        svc = svm.SVC(gamma=0.001*k, C=100)
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = model_selection.cross_validate(svc, X, y, scoring='accuracy')
        accu = scores['test_score'].mean()
        if accu >= accu_max:
            print('...')
            gamma_max = k*0.001
            accu_max = accu
    
    
    time_elapse = time.time()-time_start
    
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')
    print(f'the best accuracy gamma is: {gamma_max}, where the accuracy is: {accu_max}')
    
    return accu, time_elapse, model_name

def bays_test(X,y):
    model_name = 'bays'
    time_start=time.time()

    k_range = range(1, 31)
    k_max = 0
    accu_max = 0
    print('start training naive bays')
    #循环，取k=1到k=30，查看误差效果
    for k in k_range:
        nBays = naive_bayes.GaussianNB()
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = model_selection.cross_validate(nBays, X, y, scoring='accuracy')
        accu = scores['test_score'].mean()
        if accu >= accu_max:
            print('...')
            k_max = k
            accu_max = accu
    
    
    time_elapse = time.time()-time_start
    
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')
    print(f'the best accuracy iteration is: {k_max}, where the accuracy is: {accu_max}')
    
    return accu, time_elapse, model_name

def nn_test(X,y):
    model_name = 'nn'
    time_start=time.time()

    k_range = range(1, 31)
    k_max = 0
    accu_max = 0
    print('start training MLP')
    #循环，取k=1到k=30，查看误差效果
    for k in k_range:
        mlp = neural_network.MLPClassifier(alpha=1)
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = model_selection.cross_validate(mlp, X, y, scoring='accuracy')
        accu = scores['test_score'].mean()
        if accu >= accu_max:
            print('...')
            k_max = k
            accu_max = accu
    
    
    time_elapse = time.time()-time_start
    
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')
    print(f'the best accuracy iteration is: {k_max}, where the accuracy is: {accu_max}')
    
    return accu, time_elapse, model_name

def tree_test(X,y):
    model_name = 'tree'
    time_start=time.time()

    k_range = range(1, 31)
    k_max = 0
    accu_max = 0
    print('start training decision tree')
    #循环，取k=1到k=30，查看误差效果
    for k in k_range:
        dTree = tree.DecisionTreeClassifier(max_depth=16)
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = model_selection.cross_validate(dTree, X, y, scoring='accuracy')
        accu = scores['test_score'].mean()
        if accu >= accu_max:
            print('...')
            k_max = k
            accu_max = accu
    
    
    time_elapse = time.time()-time_start
    
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')
    print(f'the best accuracy gamma is: {k_max}, where the accuracy is: {accu_max}')
    
    return accu, time_elapse, model_name

def forest_test(X,y):
    model_name = 'forest'
    time_start=time.time()

    k_range = range(1, 31)
    k_max = 0
    accu_max = 0
    print('start training random forest')
    #循环，取k=1到k=30，查看误差效果
    for k in k_range:
        forest = ensemble.RandomForestClassifier(max_depth=16)
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = model_selection.cross_validate(forest, X, y, scoring='accuracy')
        accu = scores['test_score'].mean()
        if accu >= accu_max:
            print('...')
            k_max = k
            accu_max = accu
    
    
    time_elapse = time.time()-time_start
    
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')
    print(f'the best accuracy iteration is: {k_max}, where the accuracy is: {accu_max}')
    
    return accu, time_elapse, model_name

def adaboost_test(X,y):
    model_name = 'adaboost'
    time_start=time.time()

    k_range = range(1, 31)
    k_max = 0
    accu_max = 0
    print('start training Adaboost')
    #循环，取k=1到k=30，查看误差效果
    for k in k_range:
        adaboost = ensemble.AdaBoostClassifier()
        #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = model_selection.cross_validate(adaboost, X, y, scoring='accuracy')
        accu = scores['test_score'].mean()
        if accu >= accu_max:
            print('...')
            k_max = k
            accu_max = accu
    
    
    time_elapse = time.time()-time_start
    
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')
    print(f'the best accuracy iteration is: {k_max}, where the accuracy is: {accu_max}')
    
    return accu, time_elapse, model_name


def main():
    sample_num = 50

    X, y = datasets.make_classification(n_samples=sample_num, n_features=feature_num, n_informative=feature_num,
                        n_redundant=0, n_repeated=0, n_classes=class_num,
                        n_clusters_per_class=1, weights=None,
                        flip_y=0.01, class_sep=1.0, hypercube=True,
                        shift=0.0, scale=1.0, shuffle=True, random_state=None)
        
    accu, time_elapse, model_name = knn_test(X,y)
    accu, time_elapse, model_name = svm_test(X,y)
    accu, time_elapse, model_name = bays_test(X,y)
    accu, time_elapse, model_name = nn_test(X,y)
    accu, time_elapse, model_name = tree_test(X,y)
    accu, time_elapse, model_name = forest_test(X,y)
    accu, time_elapse, model_name = adaboost_test(X,y)

    
    
    
    
    


if __name__ == "__main__":
    main()
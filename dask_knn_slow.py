from dask_ml.datasets import make_classification
from dask_ml import model_selection
import time
import dask.array as da
import numpy as np

from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4,
                n_workers=6, memory_limit='2GB')
# client
import joblib

'''
# for real data:
    # imitating the real data
    # n_feature = 16
    #   feature1    Gender                          is 0 or 1       bool
    #   feature2    Age                             in [14, 61]     int
    #   feature3    Height                          in [1.45,1.98]  double
    #   feature4    Weight                          in [39, 173]    int
    #   feature5    family_history_with_overweight  is 0 or 1       bool
    #   feature6    FAVC                            is 0 or 1       bool
    #   feature7    FCVC                            in [1, 3]       int
    #   feature8    NCP                             in [1, 4]       int
    #   feature9    CAEC                            in [0, 3]       int
    #   feature10   SMOKE                           is 0 or 1       bool
    #   feature11   CH2O                            in [1, 3]       int
    #   feature12   SCC                             is 0 or 1       bool
    #   feature13   FAF                             in [0, 3]       int
    #   feature14   TUE                             in [0, 2]       int
    #   feature15   CALC                            in [0, 3]       int
    #   feature16   MTRANS                          in [0, 4]       int
    # n_class = 7
    #   Insufficient_Weight     0
    #   Normal_Weight           1
    #   Overweight_Level_I      2
    #   Overweight_Level_II     3
    #   Overweight_Level_I      4
    #   Obesity_Type_I          5
    #   Obesity_Type_II         6
    #   Obesity_Type_III        7
'''
feature_num = 16
class_num = 7


def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    return da.sum(y == y_pred).compute()/len(y)

def gaussian(dist, sigma = 10.0):
    """ Input a distance and return it`s weight"""
    weight = np.exp(-dist**2/(2*sigma**2))
    return weight
 
### 加权KNN
def weighted_classify(input, X_train, y_train, k):
    
    dataSize = X_train.shape[0]
    diff = da.tile(input, (dataSize, 1)) - X_train
    sqdiff = diff**2
    squareDist = np.array([sum(x) for x in sqdiff])
    dist = squareDist**0.5
    #print(input, dist[0], dist[1164])
    sortedDistIndex = np.argsort(dist)
 
    classCount = {}
    for i in range(k):
        index = sortedDistIndex[i]
        voteLabel = y_train[index].compute()
        weight = gaussian(dist[index]) 
        # print(index, dist[index],weight
        # print(classCount, voteLabel)
        ## 这里不再是加一，而是权重*1
        classCount[voteLabel] = classCount.get(voteLabel, 0) + weight*1
        
    maxCount = 0
    #print(classCount)
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
            
    return classes

class KNN:
    # based on sklearn repo on github
    # test_validation.py
    # https://github.com/scikit-learn/scikit-learn/blob/1e386a49fcaefcc9860266b5957582bc85aa56ab/sklearn/model_selection/tests/test_validation.py

    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
    
    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        return self

    def predict(self, T):
        y_pred = []
        for t in T:
            y_pred.append(weighted_classify(t, self.X, self.y, self.k))
        return y_pred

    def predict_proba(self, T):
        return T

    def score(self, X=None, Y=None):
        return 1. / (1 + abs(self.k))

    def get_params(self, deep=False):
        return {'n_neighbors': self.k}

def my_dask_knn_slow_test(X, y):
    time_start=time.time()
    test_size_split = 0.33
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size_split, shuffle=True)
    k_range = range(1, 31)
    k_max = 0
    accu_max = 0
    print('start training my dask knn slow')
    
        # 循环，取k=1到k=30，查看误差效果
    for k in k_range:
        knn = KNN(n_neighbors=k)
        with joblib.parallel_backend('dask'):
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accu = accuracy(y_test, np.array(y_pred))
            # print(f'accuracy = {accu}')
        if accu >= accu_max:
            print('...')
            k_max = k
            accu_max = accu
    print(f'the best accuracy k-value is: {k_max}, where the accuracy is: {accu_max}')


    time_end=time.time()
    time_elapse = time_end-time_start
    hours = time_elapse // 3600
    mintues = (time_elapse - hours * 3600) // 60
    seconds = time_elapse - hours * 3600 - mintues * 60
    print(f'time cost: {hours}h {mintues}m {seconds}s')

    return accu_max, time_elapse    

def main():
    sample_num = 30

    X, y = make_classification(n_samples=sample_num, n_features=feature_num,
                    n_classes=2, random_state=None, chunks=50)
    

    accu, time_elapse = my_dask_knn_slow_test(X, y)
    
    


if __name__ == "__main__":
    main()
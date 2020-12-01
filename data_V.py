from sklearn import *
from knn import *
from dask_knn import *
from dask_knn_slow import *
from ml_test import *
from dask_ml_test import *
# visualization of the data and result
import pandas as pd

from warnings import filterwarnings
filterwarnings('ignore')  # ignore warnings

dflist = []


## dist{ key:[accu, time] }
# myKnn_data = {}
# dask_myKnn_data = {}
# dask_myKnn_slow_data = {}
# knn_data = {}
# svm_data = {}
# nn_data = {}
# basys_data = {}
# tree_data = {}
# forest_data = {}
# adaboost_data = {}
# dask_knn_data = {}
# dask_svm_data = {}
# dask_nn_data = {}
# dask_basys_data = {}
# dask_tree_data = {}
# dask_forest_data = {}
# dask_adaboost_data = {}    

def test_all(sample_num):
    X, y = datasets.make_classification(n_samples=sample_num, n_features=feature_num, n_informative=feature_num,
                        n_redundant=0, n_repeated=0, n_classes=class_num,
                        n_clusters_per_class=1, weights=None,
                        flip_y=0.01, class_sep=1.0, hypercube=True,
                        shift=0.0, scale=1.0, shuffle=True, random_state=None)
    # tests all the functions except the dask_knn_slow
    accu, time_elapse, model_name = myKnn_test(X, y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = dask_myKnn_test(X, y)
    dflist.append([sample_num, model_name, time_elapse, accu])

    accu, time_elapse, model_name = knn_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = svm_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = bays_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = nn_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = tree_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = forest_test(X,y) 
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = adaboost_test(X,y) 
    dflist.append([sample_num, model_name, time_elapse, accu])

    accu, time_elapse, model_name = dask_knn_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = dask_svm_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = dask_bays_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = dask_nn_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = dask_tree_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = dask_forest_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])
    accu, time_elapse, model_name = dask_adaboost_test(X,y)
    dflist.append([sample_num, model_name, time_elapse, accu])





def main():
    # min 50

    # accu, time_elapse, model_name = dask_myKnn_slow_test(45)
    # dflist.append([45, model_name, time_elapse, accu])
    # accu, time_elapse, model_name = dask_myKnn_slow_test(50)
    # dflist.append([50, model_name, time_elapse, accu])

    # test_all(50)
    # test_all(100)
    # test_all(500)
    # test_all(1000)
    # test_all(1500)
    test_all(2000)
    
    df = pd.DataFrame(dflist, columns=['n_sample', 'model', 'seconds', 'accuracy'])
    df.to_csv("result.csv",index=False,sep=',')
    return


if __name__ == "__main__":
    main()
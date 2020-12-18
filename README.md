# CISC849Proj

This is a project that focus on analysis the parallelism performance of the Dask library for Python.

### In this project:

​	implement our own version of K Nearest Neighbor(KNN) myKnn

​	make two Dask version: 1). dask_myKnn, use Dask parallelism to speed up computation

​												2). dask_myKnn_slow, implemented with Dask array

​	call some common machine learning methods from sklearn and compare the performance with their Dask parallelism version

### The environment is: 

#### System:

​	 ![system information](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/system%20information.png)

As shown above, since my system language is Chinese, it is in Chinese. However, we can see that the machine is ASUS GX531, the CPU is i7-9750H, which has 6 cores, and the RAM is 16.0 GB. There is no GPU included in this project.

#### Python:

Python 3.7 is utilized in this project, python 2 may not be compatible

#### Library:

The imported libraries are:

​	***numpy*** for computation

​	***pandas*** for data export

​	***sklearn*** for machine learning method

​	***dask*** for parallelism

Other inclusive libraries attached from sklearn and dask can be check in the requirement text file.

### Dataset

Data portrait

In this project we are using artificial datasets so it is easier to manipulate the data size while maintain the feature.

We also retrieved a real world dataset from [**UCI Estimation of obesity levels based on eating habits and physical condition Data Set**](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+) as a paradigm and plan to apply the analysis on this data set in the future.

the obesity data looks like below:

![obesity data](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/obesity%20data.png)

it has 2112 rows, 16 features and 7 classes

the features are listed below:

![features](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/features.png)

Our artificial datasets are based on this data set, therefore, our artificial datasets have the same amount features and classes, but the data sizes are various. We are calling the make_classification function from sklearn to create datasets:

```python
sample_num = 50		# data size
feature_num = 16	# number of features
class_num = 7		# number of classes
X, y = datasets.make_classification(n_samples=sample_num, n_features=feature_num, n_informative=feature_num,
		n_redundant=0, n_repeated=0, n_classes=class_num,
		n_clusters_per_class=1, weights=None,
		flip_y=0.01, class_sep=1.0, hypercube=True,
		shift=0.0, scale=1.0, shuffle=True, random_state=None)
```



#### Dataset split

In this project, we are separating all datasets into 10 subsets, 9 for validation, and 1 for test. To achieve that, We are calling the cross_validate function from sklearn to create datasets:

```python
knn = KNN(n_neighbors=k)
scores = model_selection.cross_validate(knn, X, y, cv=10, scoring='accuracy')
```

Here KNN is an estimator we defined, there is no information about how to create an estimator, we have looked up the [test file of cross_validate](https://github.com/scikit-learn/scikit-learn/blob/255718b4ad9a3490bc99c992d467f85737bd1291/sklearn/model_selection/tests/test_validation.py) in [sklearn github](https://github.com/scikit-learn/scikit-learn) and find the example to implement our own version of KNN. **Basically, our estimator is a class, the cross_validate function is calling** *fit(self, X_subset, y_subset)* **function of the estimator first, and then the** *predict(self, X)* **function, after this ,the** *score(self, X=None, y=None)* **function.**



### How to run the code

To make it easier to run, I have also upload the virtual environment folder to this repo, therefore, reduce the incompatibility from different versions of libraries. The commands are listed below:

#### Activate the virtual environment:

```shell
venv\Scripts\activate
```

#### then run the main function.

```shell
python data_V.py
```



### File description:

***myKnn*** defines our version of KNN

***dask_myKnn*** defines the dask computation version of myKnn

***dask_myKnn_slow*** defines another dask version of myKnn that apply dask array to rewrite and runs extremely slow, but have somewhat much better accuray

***ml_test*** calls the sklearn machine learning methods 

***dask_ml_test*** utilize the dask computation of sklearn machine learning methods

***data_V*** is the main file that call all the functions to test, and save the result in "output.csv"

***result.csv*** is the sum of all outputs for analyze



### Results:

The result looks like this:

![result data](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/result%20data.png)

It shows the number of samples, the selected model, the running time in seconds, and the accuracy.

There are some left blank in result when data size is 10000 and above, because some model take too much time or too much memory, and they will not appear in the visualization.

Below are the plots generated from Excel based on the results, and we can visualize the comparison of performance.

###### dask_myKnn_slow

The time consuming bar chart of dask array version of our KNN method is shown below

![dask_myKnn_slow](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/dask_myKnn_slow%20timing.png)

###### myknn

![myKnn time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/myKnn%20timing%20comparison.png)

![myKnn accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/myKnn%20accuracy%20comparison.png)

###### KNN

![KNN time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/KNN%20timing%20comparison.png)

![KNN accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/KNN%20accuracy%20comparison.png)

###### SVM

![SVM time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/SVM%20timing%20comparison.png)

![SVM accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/SVM%20accuracy%20comparison.png)

###### Bays

![Bays time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/Naive%20Bays%20timing%20comparison.png)

![Bays accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/Naive%20Bays%20accuracy%20comparison.png)

###### MLP

![MLP time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/MLP%20timing%20comparison.png)

![MLP accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/MLP%20accuracy%20comparison.png)

###### Decision Tree

![tree time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/Decision%20Tree%20timing%20comparison.png)

![tree accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/Decision%20Tree%20accuracy%20comparison.png)

###### Random Forest

![forest time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/Random%20Forest%20timing%20comparison.png)

![forest accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/Random%20Forest%20accuracy%20comparison.png)

###### AdaBoost

![AdaBoost time](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/AdaBoost%20timing%20comparison.png)

![AdaBoost accuracy](https://github.com/ZhangQingsen/CISC849Proj/blob/main/appendix/AdaBoost%20accuracy%20comparison.png)

We can see for a few amount of computation, it is not quite that the dask parallelism version to show the advantage, while the amount of computation is larger, the performance is clearly better.


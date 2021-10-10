
# Iphone Purchase Prediction with Classification Models


## Problem Statement

The purpose of this study is; it has been estimated that users have or did not buy an iPhone based on age, salary and gender characteristics. 

## Dataset

Dataset is downloaded from [Omair Aasim's GitHub](https://github.com/omairaasim/machine_learning/tree/master/project_14_naive_bayes).  Dataset has **4 columns** and **401 rows with the header**.

## Methodology

In this project, as stated in the title, results were obtained through classifiaction methods such as ***Logistic Regression, K-Nearest Neighbors(KNN), Support Vector Classification (SVC), Decision Tree*** and ***Random Forest***.  You are free to visit [Machine Learning: Classification Models](https://medium.com/fuzz/machine-learning-classification-models-3040f71e2529) website for learn the methods better.

## Analysis


***Logistic Regression Confusion Matrix:***
| 78 | 6 |
|--|--|
| **10** | **38** |

> **Accuracy score: 0.8787878787878788**
> The model predicted as he/she get Iphone. Class:  [1]

***K-NN Confusion Matrix:***
| 78 | 6 |
|--|--|
| **6** | **42** |

> **Accuracy score: 0.9090909090909091**
> **K-NN Score: 0.9090909090909091**

The model predicted as he/she get Iphone. Class:  [1]

***SVC Confusion Matrix:***
| 78 | 6 |
|--|--|
| **13** | **35** |

> **Accuracy score: 0.8560606060606061**

The model predicted as he/she didn't get Iphone. Class:  [0]

***Decision Tree Classifier Confusion Matrix:***
| 76 | 8 |
|--|--|
| **7** | **41** |

> **Accuracy score: 0.8863636363636364**

The model predicted as he/she get Iphone. Class:  [1]

***Random Forest Confusion Matrix:***
| 77 | 7 |
|--|--|
| **4** | **44** |

> **Accuracy score: 0.9166666666666666**

The model predicted as he/she get Iphone. Class:  [1]

### AUC

**AUC Score:**
 0.9494047619047619

### K-Fold Cross Validation

**Success Mean:**
 0.8059701492537312
 
**Success Standard Deviation:**
 0.061538889934591945

### Grid Search

**Best result:**
 0.9107617051013278
 
**Best parameters:**
 {'C': 1, 'gamma': 1, 'kernel': 'rbf'}
 
> **Process took 42.99169325828552 seconds.**

## How to Run Code

Before running the code make sure that you have these libraries:

 - pandas 
 - matplotlib
 - seaborn
 - numpy
 - warnings
 - sklearn
 - time
 - mlxtend
 - pickle
    
## Contact Me

If you have something to say to me please contact me: 

 - Twitter: [Doguilmak](https://twitter.com/Doguilmak).  
 - Mail address: doguilmak@gmail.com
 

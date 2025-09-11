

# Anomaly Detection

## What is Anomaly Detection?

**Anomaly detection** is the process of identifying data points, events, or observations that deviate significantly from the expected pattern or behaviour in a dataset.

These unusual points — called **anomalies**, **outliers**, or **novelties** — may indicate critical incidents, such as:

- Fraudulent transactions
- Structural defects
- Faulty sensors
- Network intrusions
- Rare diseases



## Types of Anomaly Detection

Anomaly detection methods can be broadly classified based on the nature of the data and the context:

### 1. Point Anomalies

- **Definition:** A single data point that is significantly different from the rest.
- **Example:** A $10,000 transaction in a dataset where most transactions are under $100.

### 2. Contextual Anomalies (Conditional Anomalies)

- **Definition:** A data point that is only anomalous within a specific context.
- **Example:** A high temperature reading of 25°C may be normal in summer but anomalous in winter.

### 3. Collective Anomalies

- **Definition:** A group of data points that are anomalous when considered together, even if individual points may not be.
- **Example:** A sudden spike in server requests within a short time window could indicate a DDoS attack.


## Supervised vs. Unsupervised vs. Semi-Supervised Anomaly Detection
Anomaly detection methods can be categorized into three major approaches based on the availability of labeled data: supervised, unsupervised, and semi-supervised.

The right choice of anomaly detection technique depends on your data and the nature of the task:

* Supervised anomaly detection is ideal when you have a reliable set of labeled anomalies.
* Unsupervised anomaly detection is useful when you don’t have labeled data and can’t easily obtain it.
* Semi-supervised anomaly detection is perfect when you only have access to normal data but still need to identify deviations.

By understanding the different techniques, you can select the best method for your specific use case and dataset.

### Supervised Anomaly Detection
In supervised anomaly detection, we have labeled data that includes both normal and anomalous instances. The goal is to build a model that can classify new, unseen data points as either normal or anomalous based on the labeled training data.

When to use it:
* When labeled data is available.
* Ideal for situations where you already know what constitutes an anomaly.

Popular Algorithms:
* Logistic Regression
* Random Forest Classifier
* Support Vector Machines (SVM)
* k-Nearest Neighbors (k-NN)

### Unsupervised Anomaly Detection
Unsupervised anomaly detection doesn’t require labeled data. It assumes that anomalies are rare and different from the majority of data points. The model learns to identify anomalies based on the data’s structure, distribution, and patterns.

When to use it:
* When labeled data is unavailable or difficult to acquire.
* Ideal for scenarios where anomalies are rare, and the “normal” behaviour is well understood.

Popular Algorithms:
* Isolation Forest
* One-Class SVM
* DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
* Auto-encoders (Deep Learning)


### Semi-Supervised Anomaly Detection
Semi-supervised anomaly detection sits between supervised and unsupervised techniques. In this approach, only normal data is available for training the model. The algorithm is trained to recognize patterns in the “normal” data, and anything that deviates from that is flagged as anomalous.

When to use it:
* When only normal data is available, but you still need to detect anomalies.
* This is common when labeling anomalous data is either too expensive or impractical.

Popular Algorithms:
* One-Class SVM (used in a semi-supervised setting)
* Auto-encoders (trained only on normal data)
* k-Means Clustering (to identify normal clusters and detect outliers)

## Algorithms for Anomaly and Novelty Detection


### Fast-MCD (minimum covariance determinant)

Implemented by the EllipticEnvelope class, this algorithm is useful for outlier detection, in particular to clean up a dataset. It assumes that the normal instances (inliers) are generated from a single Gaussian distribution (not a mixture). It also assumes that the dataset is contaminated with outliers that were not generated from this Gaussian distribution. When the algorithm estimates the parameters of the Gaussian distribution (i.e., the shape of the elliptic envelope around the inliers), it is careful to ignore the instances that are most likely outliers. This technique gives a better estimation of the elliptic envelope and thus makes the algorithm better at identifying the outliers.

### Isolation forest

This is an efficient algorithm for outlier detection, especially in high-dimensional datasets. The algorithm builds a random forest in which each decision tree is grown randomly: at each node, it picks a feature randomly, then it picks a random threshold value (between the min and max values) to split the dataset in two. The dataset gradually gets chopped into pieces this way, until all instances end up isolated from the other instances. Anomalies are usually far from other instances, so on average (across all the decision trees) they tend to get isolated in fewer steps than normal instances.

### Local outlier factor (LOF)

This algorithm is also good for outlier detection. It compares the density of instances around a given instance to the density around its neighbors. An anomaly is often more isolated than its k-nearest neighbors.

### One-class SVM

This algorithm is better suited for novelty detection. Recall that a kernelized SVM classifier separates two classes by first (implicitly) mapping all the instances to a high-dimensional space, then separating the two classes using a linear SVM classifier within this high-dimensional space (see Chapter 5). Since we just have one class of instances, the one-class SVM algorithm instead tries to separate the instances in high dimensional space from the origin. In the original space, this will correspond to finding a small region that encompasses all the instances.
If a new instance does not fall within this region, it is an anomaly. There are a few hyperparameters to tweak: the usual ones for a kernelized SVM, plus a margin hyperparameter that corresponds to the probability of a new instance being mistakenly considered as novel when it is in fact normal. It works great, especially with high-dimensional datasets, but like all SVMs it does not scale to large datasets.



# References



## Books

* Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems* (3nd ed.). O’Reilly. URL: https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/



## Libraries

+ Scikit-learn
+ pycaret: https://github.com/pycaret/pycaret
+ pyod: https://github.com/LiDan456/Pyod



## Articles

* https://medium.com/data-science-collective/a-beginners-guide-to-anomaly-detection-cdebe88fc985
* Statistical Analysis with Python — Part 7 — Anomaly Detection: https://ai.plainenglish.io/statistical-analysis-with-python-part-7-anomaly-detection-120904c06fb2
* https://ai.plainenglish.io/statistical-analysis-with-python-part-7-anomaly-detection-120904c06fb2

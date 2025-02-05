# Overview 

This repository presents some Machine Learning concepts acquired from my **AI Engineering Professional Certificate specialization course**.

## Certifications

I have completed the following course:

- [Coursera: Verify my certificate](https://www.coursera.org/account/accomplishments/verify/SZ0JP21DH4Z1)

<img src="https://github.com/Mike014/LLM_Concepts/blob/main/Badge_IBM_ML.pdf" alt="Intro to Generative AI" width="300"/>

## Create a virtual environment with **Conda**:
 ```bash
 conda create -n ml_env python=3.12
 conda activate ml_env
```

### **Glossary of Terms Used in the Course**  

1. **Algorithm**: A set of step-by-step instructions for solving a problem or making a prediction in machine learning.  
2. **Argmax**: A mathematical function that returns the index of the maximum value in a set. In multi-class classification, it is used to determine the class with the highest probability.  
3. **Bagging (Bootstrap Aggregating)**: An ensemble learning method that reduces **variance** by training multiple models on random subsets of data and averaging their predictions. Used in **Random Forests**.  
4. **Bias**: Systematic error of the model, indicating how far the predictions are from the actual values. High bias causes **underfitting**, meaning the model is too simple to capture patterns in the data.  
5. **Binary Classifier**: A model that distinguishes between only two classes (e.g., positive/negative). It serves as the basis for multi-class strategies like **One-vs-All** and **One-vs-One**.  
6. **Boosting**: An ensemble learning technique that reduces **bias** by sequentially training models, each correcting the errors of the previous one. Used in **XGBoost, AdaBoost, Gradient Boosting**.  
7. **Centroid**: The central point of a cluster.  
8. **Churn Prediction**: The process of predicting whether a customer will abandon a service or subscription.  
9. **Classes**: The possible outcomes or output categories predicted by the model.  
10. **Class Imbalance**: When some classes are much more frequent than others in the dataset, influencing predictions.  
11. **Classification**: Predicting which category a piece of data belongs to by assigning it a discrete label.  
12. **Classification with KNN**: A method that assigns the class based on the majority of the K nearest neighbors.  
13. **Clustering**: Unsupervised learning technique for grouping similar data.  
14. **Convergence**: Point at which the model no longer improves, having found the minimum of the cost function.  
15. **Cosine Similarity**: Measures the angle between two vectors. Used in **NLP, Text Mining, Recommendation Systems**.  
16. **Cost Function (Log-Loss)**: Measures the difference between actual and predicted values.  
17. **Customer Segmentation**: The process of dividing customers into groups based on common characteristics.  
18. **DBSCAN**: Clustering algorithm that identifies groups based on data density.  
19. **Decision Boundary**: Line or plane that separates the classes in the dataset.  
20. **Dendrogram**: Tree diagram that represents the hierarchical structure of clustering.  
21. **Dependent Variable (Target/Output)**: The variable we want to predict (e.g. churn: yes/no).  
22. **Derivative**: Measures the rate of change of a function, used to calculate the slope of the cost function.  
23. **Distance Euclidean**: A metric for calculating the distance between two points in multidimensional space.  
24. **Distance Manhattan**: An alternative to Euclidean distance, based on orthogonal (grid-like) paths.  
25. **Dummy Class**: A fictitious class used in **One-vs-All** to separate a single class from the others.  
26. **Elbow Method**: A method for finding the optimal number of clusters in K-Means.  
27. **Epsilon-Tube**: The margin around the prediction in **SVR**, where points within the margin are not penalized.  
28. **Euclidean Distance**: Measures the linear distance between two points. Used in **K-Means, KNN, Image Analysis**.  
29. **Feature**: An independent variable used as input for the model.  
30. **Feature Irrelevant**: Useless or redundant variables that increase noise in the model and reduce accuracy.  
31. **Feature Relevant**: Input variables that help the model improve prediction.  
32. **Feature Scaling**: Feature normalization to improve model performance.  
33. **Feature Selection**: Process of choosing the most relevant features to improve model accuracy.  
34. **Feature Standardization**: The process of making features comparable, reducing their unbalanced impact on predictions.  
35. **Features**: The input (independent) variables that describe the observations.  
36. **Gamma**: A parameter of **RBF and polynomial kernels** that controls how much a single data point influences the decision boundary.  
37. **Gradient Descent**: Iterative algorithm to minimize the cost function.  
38. **Hard Margin**: Requires a perfect separation between classes, with a rigid margin.  
39. **Hierarchical Clustering**: Clustering technique that creates a hierarchical structure of groups.  
40. **Hyperplane**: A multidimensional surface that separates data into two classes.  
41. **Inference**: The process of using a trained model to make predictions on new data. The model, based on patterns learned during training, provides output for inputs it has never seen before.  
42. **Independent Variables (Feature/Input)**: The variables used to make the prediction (e.g. age, income, purchasing habits).  
43. **K-Nearest Neighbors (KNN)**: A supervised learning algorithm that uses the nearest neighbors to classify or make predictions.  
44. **K Classes**: The total number of classes in a multi-class classification problem. For example, if distinguishing between "cat," "dog," and "bird," then **K=3**.  
45. **K-Means**: Clustering algorithm that divides data into k groups based on similarity.  
46. **Kernel**: A function that transforms data to make it separable in high-dimensional spaces.  
47. **Labeled Data**: Dataset in which each example already has an assigned **class** for training.  
48. **Learning Rate (α)**: Controls the speed of parameter updates.  
49. **Linear Kernel**: Uses a simple hyperplane to separate classes.  
50. **Linear Regression**: **Regression** algorithm that predicts a continuous value based on a linear relationship between variables.  
51. **Log-Loss (Loss Function)**: Loss function used to measure the error in Logistic Regression.  
52. **Logistic Regression**: **Classification** algorithm that predicts the probability that an observation belongs to a class.  
53. **Logit**: Logarithm of the **odds ratio**, used to model log-linear relationships.  
54. **Logit Function**: Transforms any value into a probability between **0 and 1**.  
55. **Majority Voting**: A method used in **One-vs-One**, where the final classification is determined by the class that receives the most votes among all binary classifiers.  
56. **Margin**: The distance between the hyperplane and the nearest data points (**support vectors**).  
57. **Mean Squared Error (MSE)**: One of the metrics to measure the model's error.  
58. **Minkowski Distance**: Generalizes Euclidean and Manhattan distances. Used in **Clustering, KNN, Geometry**.  
59. **Multicollinearity**: Phenomenon in which two or more features are **strongly correlated**, negatively affecting the model.  
60. **Multinomial Logistic Regression**: A statistical model that generalizes binary logistic regression for multi-class classification.  
61. **Multi-Class Classification**: A problem where a given data point must be assigned to one of **K** available classes.  
62. **Observations**: are the rows in the dataset that contain information about each example.  
63. **Odds Ratio**: Ratio between the probability of success and the probability of failure.  
64. **One-Hot Encoding**: Technique to convert **categorical variables** into numeric ones (necessary for Logistic Regression).  
65. **One-vs-All (One-vs-Rest)**: A multi-class classification strategy where a binary classifier is built for each class, distinguishing it from all other classes.  
66. **One-vs-One**: A classification strategy where a binary classifier is trained for each pair of classes, and the final decision is based on the majority of votes.  
67. **Outlier Detection**: Identification of anomalous data points in a dataset.  
68. **Overfitting**: When the model is too complex and fits too closely to the training data, failing on new observations.  
69. **Parameter C**: Controls the trade-off between a stricter or softer separation in **SVM models**.  
70. **Parameters (θ)**: Model coefficients that need to be optimized.  
71. **Ponderation of Neighbors**: A technique to assign greater weight to closer neighbors in **KNN classification**.  
72. **Polynomial Kernel**: Maps data into a more complex space using polynomial functions.  
73. **Recommendation Systems**: Applications that suggest content based on clustering of users or products.  
74. **Regression**: A statistical technique that estimates a relationship between a continuous dependent variable and one or more independent variables.  
75. **Regression with KNN**: A method that assigns a numerical value by taking the **mean or median** of the values of the K nearest neighbors.  
76. **RBF (Radial Basis Function) Kernel**: Uses a transformation based on the distance between points to separate complex data.  
77. **Sigmoid**: Mathematical function that transforms the output into a value between 0 and 1.  
78. **Soft Margin**: Allows some misclassifications to improve the model's generalization.  
79. **SoftMax Probability**: The probability assigned to each class in a SoftMax model, computed by transforming the dot products of data and model parameters into a probability distribution.  
80. **SoftMax Regression**: A variant of logistic regression that assigns probabilities to multiple classes by transforming output values into a probability distribution.  
81. **Standardization of Features**: The process of making features comparable by scaling them, reducing their unbalanced impact on predictions.  
82. **Supervised Learning**: A learning method in which the model is trained on labeled data (with known features and targets).  
83. **Support Vector Machines (SVM)**: A supervised machine learning algorithm used for classification and regression.  
84. **Support Vector Regression (SVR)**: A variant of SVM for regression, which predicts continuous values.  
85. **Support Vectors**: The data points closest to the **hyperplane**, which influence the separation of classes.  
86. **Target**: The dependent or output variable that the model is intended to predict.  
87. **Theta Coefficient**: Values ​​that indicate **how much each feature affects the prediction**.  
88. **Threshold (Decision Threshold)**: The value (e.g. **0.5**) beyond which an observation is assigned to a class.  
89. **Training Set**: The dataset used to train the model.  
90. **Underfitting**: When the model is too simple and does not capture the patterns in the data well, leading to inaccurate predictions.  
91. **Unsupervised Learning**: A learning method in which the model works on **unlabeled data**, looking for patterns or structures in the dataset.  
92. **Variance**: Measures how much the model’s predictions fluctuate when trained on different subsets of the same dataset. High **variance** leads to **overfitting**.  
93. **Values of K**: The number of neighbors considered to determine the class or target value of a new point in **KNN**.

# Overview 

This repository presents some Machine Learning concepts acquired from my **AI Engineering Professional Certificate specialization course**.

## Create a virtual environment with **Conda**:
 ```bash
 conda create -n ml_env python=3.12
 conda activate ml_env
```

### **Glossary of Terms Used in the Course**  

1. **Algorithm**: A set of step-by-step instructions for solving a problem or making a prediction in machine learning.  
2. **Bagging (Bootstrap Aggregating)**: An ensemble learning method that reduces **variance** by training multiple models on random subsets of data and averaging their predictions. Used in **Random Forests**.  
3. **Bias**: Systematic error of the model, indicating how far the predictions are from the actual values. High bias causes **underfitting**, meaning the model is too simple to capture patterns in the data.  
4. **Boosting**: An ensemble learning technique that reduces **bias** by sequentially training models, each correcting the errors of the previous one. Used in **XGBoost, AdaBoost, Gradient Boosting**.  
5. **Churn Prediction**: The process of predicting whether a customer will abandon a service or subscription.  
6. **Classes**: The possible outcomes or output categories predicted by the model.  
7. **Classification**: Predicting which category a piece of data belongs to by assigning it a discrete label.  
8. **Classification with KNN**: A method that assigns the class based on the majority of the K nearest neighbors.
9. **Class Imbalance**: When some classes are much more frequent than others in the dataset, influencing predictions.  
10. **Convergence**: Point at which the model no longer improves, having found the minimum of the cost function.  
11. **Cost Function (Log-Loss)**: Measures the difference between actual and predicted values.  
12. **Decision Boundary**: Line or plane that separates the classes in the dataset.
13. **Dependent Variable (Target/Output)**: The variable we want to predict (e.g. churn: yes/no).
14. **Derivative**: Measures the rate of change of a function, used to calculate the slope of the cost function.  
15. **Distance Euclidean**: A metric for calculating the distance between two points in multidimensional space.  
16. **Distance Manhattan**: An alternative to Euclidean distance, based on orthogonal (grid-like) paths.  
17. **Epsilon-Tube**: The margin around the prediction in **SVR**, where points within the margin are not penalized.  
18. **Feature**: An independent variable used as input for the model.  
19. **Feature Irrelevant**: Useless or redundant variables that increase noise in the model and reduce accuracy.  
20. **Feature Relevant**: Input variables that help the model improve prediction.  
21. **Feature Scaling**: Feature normalization to improve model performance.
22. **Feature Selection**: Process of choosing the most relevant features to improve model accuracy.
23. **Feature Standardization**: The process of making features comparable, reducing their unbalanced impact on predictions.  
24. **Features**: The input (independent) variables that describe the observations.  
25. **Gamma**: A parameter of **RBF and polynomial kernels** that controls how much a single data point influences the decision boundary.  
26. **Gradient Descent**: Iterative algorithm to minimize the cost function.  
27. **Hard Margin**: Requires a perfect separation between classes, with a rigid margin.  
28. **Hyperplane**: A multidimensional surface that separates data into two classes.  
29. **Inference**: The process of using a trained model to make predictions on new data. The model, based on patterns learned during training, provides output for inputs it has never seen before.  
30. **Independent Variables (Feature/Input)**: The variables used to make the prediction (e.g. age, income, purchasing habits).
31. **K-Nearest Neighbors (KNN)**: A supervised learning algorithm that uses the nearest neighbors to classify or make predictions.  
32. **Kernel**: A function that transforms data to make it separable in high-dimensional spaces.  
33. **Labeled Data**: Dataset in which each example already has an assigned **class** for training.
34. **Learning Rate (α)**: Controls the speed of parameter updates.  
35. **Linear Kernel**: Uses a simple hyperplane to separate classes.  
36. **Linear Regression**: **Regression** algorithm that predicts a continuous value based on a linear relationship between variables.
37. **Log-Loss (Loss Function)**: Loss function used to measure the error in Logistic Regression.
38. **Logistic Regression**: **Classification** algorithm that predicts the probability that an observation belongs to a class.
39. **Logit**: Logarithm of the **odds ratio**, used to model log-linear relationships.
40. **Logit Function**: Transforms any value into a probability between **0 and 1**.
41. **Margin**: The distance between the hyperplane and the nearest data points (**support vectors**).  
42. **Mean Squared Error (MSE)**: One of the metrics to measure the model's error.  
43. **Multicollinearity**: Phenomenon in which two or more features are **strongly correlated**, negatively affecting the model.
44. **Observations**: are the rows in the dataset that contain information about each example.  
45. **Odds Ratio**: Ratio between the probability of success and the probability of failure.
46. **One-Hot Encoding**: Technique to convert **categorical variables** into numeric ones (necessary for Logistic Regression).
47. **Overfitting**: When the model is too complex and fits too closely to the training data, failing on new observations.  
48. **Parameter C**: Controls the trade-off between a stricter or softer separation in **SVM models**.  
49. **Parameters (θ)**: Model coefficients that need to be optimized.  
50. **Ponderation of Neighbors**: A technique to assign greater weight to closer neighbors in **KNN classification**.  
51. **Polynomial Kernel**: Maps data into a more complex space using polynomial functions.  
52. **Regression**: A statistical technique that estimates a relationship between a continuous dependent variable and one or more independent variables.  
53. **Regression with KNN**: A method that assigns a numerical value by taking the **mean or median** of the values of the K nearest neighbors.  
54. **RBF (Radial Basis Function) Kernel**: Uses a transformation based on the distance between points to separate complex data.  
55. **Sigmoid**: Mathematical function that transforms the output into a value between 0 and 1.  
56. **Soft Margin**: Allows some misclassifications to improve the model's generalization.  
57. **Standardization of Features**: The process of making features comparable by scaling them, reducing their unbalanced impact on predictions.  
58. **Supervised Learning**: A learning method in which the model is trained on labeled data (with known features and targets).  
59. **Support Vector Machines (SVM)**: A supervised machine learning algorithm used for classification and regression.  
60. **Support Vector Regression (SVR)**: A variant of SVM for regression, which predicts continuous values.  
61. **Support Vectors**: The data points closest to the **hyperplane**, which influence the separation of classes.  
62. **Target**: The dependent or output variable that the model is intended to predict.  
63. **Theta Coefficient**: Values ​​that indicate **how much each feature affects the prediction**.
64. **Threshold (Decision Threshold)**: The value (e.g. **0.5**) beyond which an observation is assigned to a class.
65. **Training Set**: The dataset used to train the model.  
66. **Underfitting**: When the model is too simple and does not capture the patterns in the data well, leading to inaccurate predictions.  
67. **Unsupervised Learning**: A learning method in which the model works on **unlabeled data**, looking for patterns or structures in the dataset.  
68. **Variance**: Measures how much the model’s predictions fluctuate when trained on different subsets of the same dataset. High **variance** leads to **overfitting**.  
79. **Values of K**: The number of neighbors considered to determine the class or target value of a new point in **KNN**.



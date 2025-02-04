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
9. **Decision Boundary**: Line or plane that separates the classes in the dataset.
10. **Dependent Variable (Target/Output)**: The variable we want to predict (e.g. churn: yes/no).
11. **Distance Euclidean**: A metric for calculating the distance between two points in multidimensional space.  
12. **Distance Manhattan**: An alternative to Euclidean distance, based on orthogonal (grid-like) paths.  
13. **Epsilon-Tube**: The margin around the prediction in **SVR**, where points within the margin are not penalized.  
14. **Feature Irrelevant**: Useless or redundant variables that increase noise in the model and reduce accuracy.  
15. **Feature Relevant**: Input variables that help the model improve prediction.  
16. **Feature Scaling**: Feature normalization to improve model performance.
17. **Feature Selection**: Process of choosing the most relevant features to improve model accuracy.
18. **Feature Standardization**: The process of making features comparable, reducing their unbalanced impact on predictions.  
19. **Features**: The input (independent) variables that describe the observations.  
20. **Gamma**: A parameter of **RBF and polynomial kernels** that controls how much a single data point influences the decision boundary.  
21. **Hard Margin**: Requires a perfect separation between classes, with a rigid margin.  
22. **Hyperplane**: A multidimensional surface that separates data into two classes.  
23. **Inference**: The process of using a trained model to make predictions on new data. The model, based on patterns learned during training, provides output for inputs it has never seen before.  
24. **Independent Variables (Feature/Input)**: The variables used to make the prediction (e.g. age, income, purchasing habits).
25. **K-Nearest Neighbors (KNN)**: A supervised learning algorithm that uses the nearest neighbors to classify or make predictions.  
26. **Kernel**: A function that transforms data to make it separable in high-dimensional spaces.  
27. **Labeled Data**: Dataset in which each example already has an assigned **class** for training.
28. **Linear Kernel**: Uses a simple hyperplane to separate classes.  
29. **Linear Regression**: **Regression** algorithm that predicts a continuous value based on a linear relationship between variables.
30. **Log-Loss (Loss Function)**: Loss function used to measure the error in Logistic Regression.
31. **Logistic Regression**: **Classification** algorithm that predicts the probability that an observation belongs to a class.
32. **Logit**: Logarithm of the **odds ratio**, used to model log-linear relationships.
33. **Logit Function**: Transforms any value into a probability between **0 and 1**.
34. **Margin**: The distance between the hyperplane and the nearest data points (**support vectors**).  
35. **MSE (Mean Squared Error)**: Measures how close the target values at a node are to the mean, commonly used in **regression models**.  
36. **Multicollinearity**: Phenomenon in which two or more features are **strongly correlated**, negatively affecting the model.
37. **Observations**: are the rows in the dataset that contain information about each example.  
38. **Odds Ratio**: Ratio between the probability of success and the probability of failure.
39. **One-Hot Encoding**: Technique to convert **categorical variables** into numeric ones (necessary for Logistic Regression).
40. **Overfitting**: When the model is too complex and fits too closely to the training data, failing on new observations.  
41. **Parameter C**: Controls the trade-off between a stricter or softer separation in **SVM models**.  
42. **Ponderation of Neighbors**: A technique to assign greater weight to closer neighbors in **KNN classification**.  
43. **Polynomial Kernel**: Maps data into a more complex space using polynomial functions.  
44. **Regression**: A statistical technique that estimates a relationship between a continuous dependent variable and one or more independent variables.  
45. **Regression with KNN**: A method that assigns a numerical value by taking the **mean or median** of the values of the K nearest neighbors.  
46. **RBF (Radial Basis Function) Kernel**: Uses a transformation based on the distance between points to separate complex data.  
47. **Soft Margin**: Allows some misclassifications to improve the model's generalization.  
48. **Squilibrio nelle classi (Class Imbalance)**: When some classes are much more frequent than others in the dataset, influencing predictions.  
49. **Standardization of Features**: The process of making features comparable by scaling them, reducing their unbalanced impact on predictions.  
50. **Supervised Learning**: A learning method in which the model is trained on labeled data (with known features and targets).  
51. **Support Vector Machines (SVM)**: A supervised machine learning algorithm used for classification and regression.  
52. **Support Vector Regression (SVR)**: A variant of SVM for regression, which predicts continuous values.  
53. **Support Vectors**: The data points closest to the **hyperplane**, which influence the separation of classes.  
54. **Target**: The dependent or output variable that the model is intended to predict.  
55. **Theta Coefficient**: Values ​​that indicate **how much each feature affects the prediction**.
56. **Threshold (Decision Threshold)**: The value (e.g. **0.5**) beyond which an observation is assigned to a class.
57. **Underfitting**: When the model is too simple and does not capture the patterns in the data well, leading to inaccurate predictions.  
58. **Unsupervised Learning**: A learning method in which the model works on **unlabeled data**, looking for patterns or structures in the dataset.  
59. **Variance**: Measures how much the model’s predictions fluctuate when trained on different subsets of the same dataset. High **variance** leads to **overfitting**.  
60. **Values of K**: The number of neighbors considered to determine the class or target value of a new point in **KNN**.



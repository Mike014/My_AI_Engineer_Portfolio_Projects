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
9. **Distance Euclidean**: A metric for calculating the distance between two points in multidimensional space.  
10. **Distance Manhattan**: An alternative to Euclidean distance, based on orthogonal (grid-like) paths.  
11. **Epsilon-Tube**: The margin around the prediction in **SVR**, where points within the margin are not penalized.  
12. **Feature Irrelevant**: Useless or redundant variables that increase noise in the model and reduce accuracy.  
13. **Feature Relevant**: Input variables that help the model improve prediction.  
14. **Feature Standardization**: The process of making features comparable, reducing their unbalanced impact on predictions.  
15. **Features**: The input (independent) variables that describe the observations.  
16. **Gamma**: A parameter of **RBF and polynomial kernels** that controls how much a single data point influences the decision boundary.  
17. **Hard Margin**: Requires a perfect separation between classes, with a rigid margin.  
18. **Hyperplane**: A multidimensional surface that separates data into two classes.  
19. **Inference**: The process of using a trained model to make predictions on new data. The model, based on patterns learned during training, provides output for inputs it has never seen before.  
20. **K-Nearest Neighbors (KNN)**: A supervised learning algorithm that uses the nearest neighbors to classify or make predictions.  
21. **Kernel**: A function that transforms data to make it separable in high-dimensional spaces.  
22. **Labeled Data**: Input data that includes the correct answer (label or class) for training.  
23. **Linear Kernel**: Uses a simple hyperplane to separate classes.  
24. **Margin**: The distance between the hyperplane and the nearest data points (**support vectors**).  
25. **MSE (Mean Squared Error)**: Measures how close the target values at a node are to the mean, commonly used in **regression models**.  
26. **Observations**: are the rows in the dataset that contain information about each example.  
27. **Overfitting**: When the model is too complex and fits too closely to the training data, failing on new observations.  
28. **Parameter C**: Controls the trade-off between a stricter or softer separation in **SVM models**.  
29. **Polynomial Kernel**: Maps data into a more complex space using polynomial functions.  
30. **Ponderation of Neighbors**: A technique to assign greater weight to closer neighbors in **KNN classification**.  
31. **Regression**: A statistical technique that estimates a relationship between a continuous dependent variable and one or more independent variables.  
32. **Regression with KNN**: A method that assigns a numerical value by taking the **mean or median** of the values of the K nearest neighbors.  
33. **RBF (Radial Basis Function) Kernel**: Uses a transformation based on the distance between points to separate complex data.  
34. **Soft Margin**: Allows some misclassifications to improve the model's generalization.  
35. **Squilibrio nelle classi (Class Imbalance)**: When some classes are much more frequent than others in the dataset, influencing predictions.  
36. **Standardization of Features**: The process of making features comparable by scaling them, reducing their unbalanced impact on predictions.  
37. **Supervised Learning**: A learning method in which the model is trained on labeled data (with known features and targets).  
38. **Support Vector Machines (SVM)**: A supervised machine learning algorithm used for classification and regression.  
39. **Support Vector Regression (SVR)**: A variant of SVM for regression, which predicts continuous values.  
40. **Support Vectors**: The data points closest to the **hyperplane**, which influence the separation of classes.  
41. **Target**: The dependent or output variable that the model is intended to predict.  
42. **Underfitting**: When the model is too simple and does not capture the patterns in the data well, leading to inaccurate predictions.  
43. **Unsupervised Learning**: A learning method in which the model works on **unlabeled data**, looking for patterns or structures in the dataset.  
44. **Variance**: Measures how much the model’s predictions fluctuate when trained on different subsets of the same dataset. High **variance** leads to **overfitting**.  
45. **Values of K**: The number of neighbors considered to determine the class or target value of a new point in **KNN**.  



### **Linear Algebra Glossary for Deep Learning (Python Approach)**  

---

### **Broadcasting**  
- Automatic extension of array dimensions to enable operations between arrays of different shapes.  
- Example: Adding a vector to a matrix where the vector is **implicitly copied** across rows.  
- **Python**: `A + b` (adds `b` to each row of `A`).  

---

### **Indices**  
- Integers that **identify positions** within a vector, matrix, or tensor.  
- Example: `A[i, j]` accesses the element at row `i`, column `j` in matrix `A`.  

---

### **Matrix**  
- A **2D array of numbers** with rows and columns, used to store and manipulate data in machine learning models.  
- **Python (NumPy)**: `np.array([[A11, A12], [A21, A22]])`.  

---

### **NumPy**  
- A Python library for numerical computations.  
- Provides efficient operations on **scalars, vectors, matrices, and tensors**.  
- Essential for deep learning frameworks like TensorFlow and PyTorch.  

---

### **Scalar**  
- A **single numerical value**, the simplest data type in linear algebra.  
- Examples: `5`, `3.14`.  
- **Python**: `int`, `float`.  

---

### **Tensor**  
- A **generalization of matrices** to **higher dimensions** (multidimensional arrays).  
- Used in deep learning for batch processing and feature representation.  
- **Example (3D tensor)**: `np.random.rand(3, 3, 3)`.  

---

### **Transpose**  
- **Swaps rows and columns** in a matrix, often used in gradient computations and optimizations.  
- **Mathematical Notation**: \( A^T \).  
- **Python**: `A.T`.  

---

### **Vector**  
- A **1D array** representing a list of numbers or coordinates in space.  
- Used to store weights, biases, and input features in machine learning models.  
- **Python (NumPy)**: `np.array([x1, x2, x3])`.  

---

### **Element-wise Operations**  
- Operations applied **independently** to each corresponding element in two or more arrays.  
- Example: **Hadamard Product** (element-wise multiplication).  
- **Mathematical Notation**: \( C = A \odot B \).  
- **Python**: `C = A * B`.  

---

### **Matrix Multiplication**  
- **Transforms data** by combining multiple features using weights.  
- Essential in **neural networks** for forward propagation.  
- **Mathematical Notation**:  
  \[
  $C_{i,j} = \sum_k A_{i,k} B_{k,j}$
  \]
- **Python**: `C = A @ B` or `np.dot(A, B)`.  

---

### **Dot Product of Two Vectors**  
- **Computes similarity** between two vectors, used in optimization and feature selection.  
- **Mathematical Notation**:  
  \[
  $x^T y = \sum_i x_i y_i$
  \]
- **Python**: `np.dot(x, y)`.  

---

### **Solving Linear Systems (Ax = b)**  
- Used to compute **unknown variables** in machine learning models.  
- Essential for **linear regression and optimization**.  
- **Mathematical Notation**: \( Ax = b \).  
- **Python**: `np.linalg.solve(A, b)`.  


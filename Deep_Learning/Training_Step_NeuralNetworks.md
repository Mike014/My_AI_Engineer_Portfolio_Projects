### **Neural Network Training Process**  

**The key concept is continuous iteration** to improve the model until it reaches an acceptable error level.  

1️⃣ **Forward Propagation**  
   - Data is fed through the network.  
   - Weights and biases determine the network’s output.  
   - The network generates a prediction.  

2️⃣ **Error Calculation**  
   - The network’s output is compared to the actual value (**ground truth**).  
   - The error is computed using a cost function.  

3️⃣ **Backward Propagation**  
   - The error is propagated backward through the network.  
   - The gradient (derivative of the cost function) is calculated.  
   - Weights and biases are updated using **gradient descent**.  

4️⃣ **Iteration**  
   - The cycle **Forward → Error → Backward → Update** is repeated.  
   - This continues for a certain number of **epochs** or until the error is sufficiently small.  

5️⃣ **Target Achieved**  
   - When the error reaches its minimum, the model is ready to make accurate predictions!  

In summary, it is an **iterative process** where each cycle improves the model until we achieve an **output close to reality**.
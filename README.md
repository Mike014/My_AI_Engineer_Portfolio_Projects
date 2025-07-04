# Overview 

This repository presents some [Machine Learning](https://developers.google.com/machine-learning/advanced-courses?hl=it) concepts acquired from my **AI Engineering Professional Certificate specialization course**.

Here is a brief description of the libraries I have worked with so far in my AI Engineering journey:

- **TensorFlow**: Google's open-source library for Machine Learning and Deep Learning. It supports complex neural networks and optimized computations on GPUs and TPUs.

- **Keras**: High-level API integrated into TensorFlow, designed for building and training Deep Learning models in an intuitive and modular way.

- **SmolAgents** (by Hugging Face): Framework for building autonomous AI agents, allowing you to combine LLM with interactive environments and automation tools.

- **NLTK** (Natural Language Toolkit): Library for natural language processing (NLP), with tools for tokenization, stemming, lemmatization, parsing and more.

- **Hugging Face Transformers**: Library for working with advanced NLP models such as BERT, GPT, T5 and LLaMA. It allows fine-tuning and inference on pre-trained models.

- **Stable-Baselines3**: Advanced implementation of **Reinforcement Learning** algorithms based on PyTorch, used to train AI agents in various simulated and real environments.

## Certifications

I have completed the following course:

- [Coursera: Verify my certificate](https://www.coursera.org/account/accomplishments/verify/SZ0JP21DH4Z1)

<img src="https://github.com/Mike014/My_AI_Engineer_Portfolio_Projects/blob/main/Badge_IBM_ML.png" alt="Intro to Generative AI" width="300"/>

## Create a virtual environment with **Conda**:
 ```bash
 conda create -n ml_env python=3.12
 conda activate ml_env
```

### **MAP OF AI**

- **𝗔𝗿𝘁𝗶𝗳𝗶𝗰𝗶𝗮𝗹 𝗜𝗻𝘁𝗲𝗹𝗹𝗶𝗴𝗲𝗻𝗰𝗲 (𝗔𝗜)** – The broadest category, covering automation, reasoning, and decision-making. Early AI was rule-based, but today, it's mainly data-driven.
- **𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 (𝗠𝗟)** – AI that learns patterns from data without explicit programming. Includes decision trees, clustering, and regression models.
- **𝗡𝗲𝘂𝗿𝗮𝗹 𝗡𝗲𝘁𝘄𝗼𝗿𝗸𝘀 (𝗡𝗡)** – A subset of ML, inspired by the human brain, designed for pattern recognition and feature extraction.
- **𝗗𝗲𝗲𝗽 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 (𝗗𝗟)** – Multi-layered neural networks that drives a lot of modern AI advancements, for example enabling image recognition, speech processing, and more.
- **𝗧𝗿𝗮𝗻𝘀𝗳𝗼𝗿𝗺𝗲𝗿𝘀** – A revolutionary deep learning architecture introduced by Google in 2017 that allows models to understand and generate language efficiently.
- **𝗚𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝘃𝗲 𝗔𝗜 (𝗚𝗲𝗻𝗔𝗜)** – AI that doesn’t just analyze data—it creates. From text and images to music and code, this layer powers today's most advanced AI models.
- **𝗚𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝘃𝗲 𝗣𝗿𝗲-𝗧𝗿𝗮𝗶𝗻𝗲𝗱 𝗧𝗿𝗮𝗻𝘀𝗳𝗼𝗿𝗺𝗲𝗿𝘀 (𝗚𝗣𝗧)** – A specific subset of Generative AI that uses transformers for text generation.
- **𝗟𝗮𝗿𝗴𝗲 𝗟𝗮𝗻𝗴𝘂𝗮𝗴𝗲 𝗠𝗼𝗱𝗲𝗹𝘀 (𝗟𝗟𝗠)** – Massive AI models trained on extensive datasets to understand and generate human-like language.
- **𝗚𝗣𝗧-4** – One of the most advanced LLMs, built on transformer architecture, trained on vast datasets to generate human-like responses.
- **𝗖𝗵𝗮𝘁𝗚𝗣𝗧** – A specific application of GPT-4, optimized for conversational AI and interactive use.

## **What is Artificial Intelligence (AI)?**
**Artificial Intelligence (AI)** is a branch of computer science focused on creating systems capable of simulating aspects of human intelligence. This includes activities such as learning, reasoning, perception, language comprehension, and problem-solving.

AI is based on algorithms and mathematical models that allow machines to **analyze data, recognize patterns, and make decisions** autonomously or with human assistance.

---

## **1. Types of Artificial Intelligence**
AI can be divided into **three main categories** based on the level of autonomy and reasoning capability.

### **A. Weak AI (Narrow AI)**
This is the current form of AI, designed to perform **specific tasks**. It does not have general intelligence like a human.
- **Examples**: Voice assistants (Siri, Alexa), recommendation engines (Netflix, Spotify), chatbots, facial recognition.

### **B. Strong AI (General AI)**
A hypothetical AI that would be capable of **thinking and learning like a human** autonomously.
- **Examples**: It does not exist yet, but it would be an AI capable of adapting to any intellectual task without requiring specific training for each.

### **C. Super Artificial Intelligence (Super AI)**
A theoretical level where AI would surpass human intelligence in all fields, including creativity, problem-solving, and intuition.
- **Examples**: This does not yet exist but is the basis of many theories about AI’s future.

---

## **2. Main AI Fields**
AI is a vast field that encompasses many disciplines. Here are the primary ones:

### **A. Machine Learning (ML) – Automatic Learning**
Machine Learning is a subfield of AI that allows machines to **learn from data** without explicit programming.
- **Main types of ML**:
  - **Supervised Learning** → AI learns from labeled data (e.g., image recognition with predefined descriptions).
  - **Unsupervised Learning** → AI finds patterns in data without human guidance (e.g., user clustering on streaming platforms).
  - **Reinforcement Learning** → AI improves its decisions by receiving rewards or penalties (e.g., AI in games like AlphaGo).

### **B. Natural Language Processing (NLP)**
Allows AI to **understand, analyze, and generate text or speech**, enabling human-machine interaction.
- **Examples**: ChatGPT, automatic translators, sentiment analysis in social media.

### **C. Computer Vision**
Enables AI to **analyze images and videos**, recognizing objects, faces, or scenes.
- **Examples**: Facial recognition, AI-powered medical diagnostics, self-driving cars.

### **D. Generative AI**
Allows AI to **create text, images, music, and videos** autonomously.
- **Examples**: 
  - **Text**: GPT-4 for automated writing.
  - **Images**: DALL·E, Stable Diffusion for creating digital art.
  - **Music**: OpenAI’s Jukebox for generating music tracks.
  - **Video**: AI models capable of generating realistic videos from scratch.

### **E. Robotics**
AI is applied to robotics to **create machines capable of moving and interacting with the real world**.
- **Examples**: Humanoid robots (Boston Dynamics), industrial robotic arms, Mars exploration robots.

---

## **3. How Does an AI Model Work?**
AI operates through three main phases:

1. **Data Collection** → AI needs data to learn (texts, images, sounds, etc.).
2. **Model Training** → An algorithm analyzes data and looks for patterns to make predictions or decisions.
3. **Application and Optimization** → The model is applied in the real world and continuously improved.

---

## **4. Challenges and Ethical Issues in AI**
AI also presents **ethical challenges and risks**, including:

- **Bias and Discrimination** → If AI is trained on biased data, it can produce unfair outcomes (e.g., facial recognition accuracy varying by ethnicity).
- **Job Loss** → Automation may replace certain human jobs.
- **Privacy and Security** → AI can collect and use personal data in intrusive ways.
- **Manipulation and Fake News** → Generative AI can be used to create misleading or false content.

---

## **5. The Future of Artificial Intelligence**
AI will continue to evolve and impact more sectors, including:

- **Healthcare** → More accurate diagnostics with AI.
- **Industrial Automation** → Autonomous robots for manufacturing.
- **Entertainment** → AI in video games, movies, and music.
- **Sustainability** → AI optimizing energy consumption and reducing pollution.

AI is a **powerful, revolutionary, and continuously evolving** technology, shaping the modern world.

### **Artificial Intelligence (AI) Overview**  
A concise overview of **Artificial Intelligence (AI)** with its main categories and subcategories.

---

## **1. Artificial Intelligence (AI)**  
AI is a field of computer science that develops systems capable of simulating human cognitive processes.

### **Main Categories:**  
- **Weak AI (Narrow AI)** → Specialized in specific tasks (e.g., voice assistants, chatbots).  
- **Strong AI (General AI)** → A theoretical AI capable of learning and adapting like a human.  
- **Super AI** → A hypothetical form of AI that surpasses human intelligence.  

---

## **2. Types of AI (Approaches)**  
### **A. Machine Learning (ML)**  
AI that **learns from data** without explicit programming.  

#### **ML Subcategories:**  
- **Supervised Learning** → Learns from labeled data.  
- **Unsupervised Learning** → Finds patterns in unlabeled data.  
- **Reinforcement Learning** → Learns through rewards and penalties.  

### **B. Deep Learning (DL)**  
An advanced subset of ML that uses **deep neural networks**.  

#### **DL Subcategories (Neural Networks):**  
- **Feedforward Neural Networks (FNN)** → Basic structure for pattern recognition.  
- **Convolutional Neural Networks (CNN)** → Used for computer vision (image analysis).  
- **Recurrent Neural Networks (RNN)** → Ideal for sequential data (text, audio).  
- **Transformers** → Evolution of RNNs, used for NLP (ChatGPT, BERT).  

---

## **3. Other AI Branches**  
### **A. Natural Language Processing (NLP)**  
AI for **understanding and generating human language**.  
- **Examples**: Chatbots, automatic translation, sentiment analysis.  

### **B. Computer Vision**  
AI that analyzes **images and videos**.  
- **Examples**: Facial recognition, self-driving cars, medical diagnostics.  

### **C. Generative AI**  
AI that **creates new content** (text, images, music, video).  
- **Examples**: DALL·E (images), GPT (text), Jukebox (music).  

### **D. Robotics & Autonomous Systems**  
AI applied to **robots and autonomous systems**.  
- **Examples**: Industrial robots, self-driving vehicles.  

### **E. Expert Systems**  
AI based on **predefined logical rules**, without learning.  
- **Examples**: Medical diagnosis systems using symptom databases.  

### **F. Edge AI**  
AI that **processes data directly on devices**, without relying on the cloud.  
- **Examples**: Offline voice assistants, AI in smartphones.  

---

## **4. AI Challenges and Issues**  
- **Bias and discrimination** in data.  
- **Privacy and security** in data usage.  
- **AI ethics** and regulatory concerns.  
- **Automation and its impact on jobs.**  

---

### **Conclusion**  
AI is a vast field with many specializations. **Machine Learning** and **Deep Learning** are fundamental, but other crucial areas include **NLP, Computer Vision, and Generative AI**.

### **Glossary of [Google](https://developers.google.com/machine-learning/glossary?hl=it#a)**

### **Glossary of Terms Used in the Course** 

- **Activation Function**: A mathematical function that introduces **non-linearity** into the model, allowing it to solve complex problems.  
- **Adam Optimizer**: An advanced optimization algorithm that improves gradient descent.  
- **Algorithm**: A set of step-by-step instructions for solving a problem or making a prediction in machine learning.  
- **Algorithm design**: The **laws of probability guide how AI systems should reason**. Algorithms are therefore built to compute (or approximate) probabilistic expressions.
- **Argmax**: A function that returns the index of the maximum value in a set. In multi-class classification, it determines the class with the highest probability.  
- **Artificial Neuron**: A mathematical model inspired by biological neurons, used in artificial neural networks.  
- **Autoencoder**: An unsupervised neural network that learns to compress and decompress data without human intervention.  
- **Automatic Translation**: The use of neural networks to translate text between different languages.  
- **Average-Pooling**: Computes the average value within a region of the image.  
- **Autoregression**: In an autoregressive model, the output generated in a given step is used as input for the next step. This creates a **sequential dependency**.
- **Backpropagation**: An algorithm that optimizes **weights** and **biases** by correcting the model’s errors.  
- **BatchNormalization (Normalization)**: Technique that improves training stability and speed by **normalizing the output of layers to have a mean of zero** and a variance of one, applied during both training and inference.
- **Batch Size**: Number of samples processed by the model in a single training step.
- **Bagging (Bootstrap Aggregating)**: An ensemble learning method that reduces **variance** by training multiple models on random subsets of data and averaging their predictions. Used in **Random Forests**.
- **Bellman Equation Formula**: used to update Q-values ​​by leveraging the reward and future Q-value estimation. 
- **Bias**: Systematic error of the model, indicating how far the predictions are from the actual values. High bias causes **underfitting**.  
- **Bias (b)**: A constant added to the equation to improve the model’s learning ability.  
- **Binary Classifier**: A model that distinguishes between only two classes (e.g., positive/negative). It serves as the basis for multi-class strategies like **One-vs-All** and **One-vs-One**.  
- **Boosting**: An ensemble learning technique that reduces **bias** by sequentially training models, each correcting the errors of the previous one. Used in **XGBoost, AdaBoost, Gradient Boosting**.
- **Broadcasting**: A rule that allows operations between arrays of different sizes by automatically adjusting dimensions. 
- **Centroid**: The central point of a cluster.  
- **Categorical Cross-Entropy**: A loss function used for multi-class classification.  
- **CartPole**: Classic RL environment where you try to balance a pole on a moving cart.
- **Cost Function**: It would calculate how wrong the model's answer is.
- **Checkerboard Artifacts**: Distortions due to uneven overlapping of convolutional filters.
- **Churn Prediction**: The process of predicting whether a customer will abandon a service or subscription.  
- **Classes**: The possible outcomes or output categories predicted by the model.  
- **Class Imbalance**: When some classes are much more frequent than others in the dataset, influencing predictions.  
- **Classification**: Predicting which category a piece of data belongs to by assigning it a discrete label.  
- **Classification with KNN**: A method that assigns the class based on the majority vote of the K nearest neighbors.  
- **Clustering**: An unsupervised learning technique for grouping similar data.  
- **CNN (Convolutional Neural Network)**: A neural network excellent for processing images and static objects, though it does not consider temporal context. 
- **Convolution operation in CNN**: A mathematical process that allows the network to extract relevant features from an image.
- **Convolutional Base**: Part of a CNN model that extracts features from images without including fully connected layers. 
- **Conv2D (Convolutional Layer)**: A convolutional layer that applies filters to the input to extract important features.
- **Cosine Similarity**: how similar two vectors are
- **Cropping2D (Output Cropping)**: Trims parts of the output to correct mismatches in dimensions.
- **K-means algorithm**: A popular clustering technique that partitions a dataset into distinct groups based on the features of the data points.
- **Data Augmentation**: is a pre-processing technique used in deep learning to artificially increase the amount of data available for training a model. This is done by applying random transformations to existing images, creating new versions with different variations.
- **Data Denoising**: Automatic noise removal from data via autoencoders.  
- **Decoder**: The part of an autoencoder that reconstructs the original input from the compressed representation. Generate the output sequence, step by step, using the encoder representation and previously generated information.
- **Deep Reinforcement Learning (Deep RL)**: is a **subset of Machine Learning** in which an **agent learns to behave in an environment** by **performing actions** and **observing the results**.
- **Deep Neural Network (DNN)**: A neural network with three or more hidden layers, capable of processing raw data such as images and text.  
- **Dependent Variable (Target/Output)**: The variable that the model is intended to predict (e.g., churn: yes/no).  
- **Denoising**: The process of removing noise from data, such as images, to improve their quality.
- **Dimensionality Reduction**: A technique to reduce the number of features in the data, improving efficiency and interpretability.  
- **Derivative**: Measures the rate of change of a function; used to calculate the slope of the cost function.  
- **Diffusion models***: A **class of generative machine learning models** that produce **high-quality synthetic data**. These models work by gradually **transforming a noisy initial sample** through a **series of passes**.
- **Discount Factor (γ)**: Discount factor that penalizes future rewards compared to immediate ones.
- **Distance Euclidean**: A metric for calculating the distance between two points in multidimensional space.  
- **Distance Manhattan**: A metric based on orthogonal (grid-like) paths, an alternative to Euclidean distance.  
- **DQN (Deep Q-Network)**: Variant that uses a neural network instead of the Q-table.
- **Dropout**: Regularization technique that **reduces overfitting** by randomly deactivating a fraction of neurons during training.
- **Dummy Class**: A fictitious class used in **One-vs-All** to separate a single class from the others.  
- **Elastic Weight Consolidation (EWC)**: A **biologically inspired algorithm** that **slows learning on critical weights** for previous tasks, **preserving past knowledge** while learning new tasks.
- **Elbow Method**: A method for finding the optimal number of clusters in K-Means. 
- **Embedding Layer**: is a **way to represent categorical or discrete data** as continuous **vectors**. 
- **Encoder**: The part of an autoencoder that reduces the dimensionality of data into a more compact representation. 
Transforms the input sequence into a dense numerical representation, capturing meaning and context.
- **Environment:** An environment is a simulated world where an agent can learn by interacting with it.
- **Epsilon-Tube**: The margin around the prediction in **SVR**, within which points are not penalized.
- **Epsilon-greedy Strategy**: that balances exploration and exploitation: acts randomly with probability epsilon, otherwise follows the policy.  
- **Epoch**: A complete training cycle where the model has seen all input data once.  
- **Experience Replay**: A technique that saves past experiences in a buffer and samples them randomly to stabilize learning.
- **Exploration Rate (ε)** Parameter that controls the probability of random choice of the action.
- **Euclidean Distance**: Measures the straight-line distance between two points. Used in **K-Means, KNN, Image Analysis**.  
- **Feature**: An independent variable used as input for the model.  
- **Feature Extraction**: A process in which a **pre-trained neural network** is used to extract significant features from new images.
- **Feature Map**: The output of a convolutional layer representing learned patterns from input data.
- **Feature Irrelevant**: Useless or redundant variables that increase noise in the model and reduce accuracy.  
- **Feature Relevant**: Input variables that help the model improve prediction accuracy.  
- **Feature Scaling**: The process of normalizing features to improve model performance.  
- **Feature Selection**: The process of choosing the most relevant features to improve model accuracy.  
- **Feature Standardization**: The process of scaling features so that they are comparable, reducing their unbalanced impact on predictions.  
- **Features**: The input (independent) variables that describe the observations.  
- **Feed-Forward Neural Network (FFNN)**: A neural network transforms the processed vector. **Feedforward Network**,  connections between nodes **do not form cycles**, **information** only moves in one direction, **forward**.
- **Fine-Tuning**: A technique in which some layers of a pre-trained model are "unlocked" and retrained on a new specific dataset.
- **Fisher Information**: Fisher Information tells you how "sensitive" the probability of your data is to small changes in these parameters.
- **Flattening**: Transforming the convolutional output into a vector for the dense layer.  
- **Forward Propagation**: The process by which data passes through the **neural network**, from input to output.  

- **Fully Connected Layer**: Also known as the **Dense Layer**; the final layer for classification using **Softmax**. The **final layers** of a CNN that perform classification.
- **Functional API**: An alternative to the Sequential API that allows for creating more **complex and flexible** models, with multiple inputs/outputs and non-linear connections.
- **Gamma**: A parameter of **RBF and polynomial kernels** that controls how much a single data point influences the decision boundary.  
- **Gym (OpenAI) Toolkit**: for simulating standardized RL environments like CartPole, Pong, etc.
- **Gradient Descent**: An iterative algorithm used to minimize the cost function.  
- **Gradients**: Values that indicate how much the network's weights should be updated. Too small values make learning slow.  
- **Ground Truth**: The actual or correct value that the model is intended to predict.  
- **Hard Margin**: A requirement for perfect separation between classes with a rigid margin.  .
- **Hebb's theory**: simply put, says that neurons that fire together strengthen their connections, "neurons that fire together, wire together."
- **Hidden Layer**: An intermediate layer in the neural network that processes information.  
- **Hidden Layers**: Multiple intermediate layers that process data between the input and output layers.  
- **Hierarchical Clustering**: A clustering technique that creates a hierarchical structure of groups. 
- **Hyperparameter**: These are model values ​​or settings that are not automatically learned from the data, but are manually chosen by the researcher, or optimized with particular search strategies. **Hyperparameters are values that define how your model is trained**
- **Hyperparameter Tuning**: is the process of selecting the **optimal values** for a machine learning model’s hyperparameters. 
- **Hyperbolic Tangent (Tanh)**: A sigmoid variant with outputs ranging from -1 to 1, providing more balanced values.  
- **Hyperplane**: A multidimensional surface that separates data into different classes.  
- **Image Classification**: An application of neural networks where images are categorized into different classes.  
- **Independent Variables (Feature/Input)**: The variables used for making predictions (e.g., age, income, purchasing habits).  
- **Inference**: The process of using a trained model to make predictions on new, unseen data.  
- **Information theory**: Is used to **quantify uncertainty** within a probability distribution.
- **Input Layer**: The first layer of the neural network that receives the initial data.  
- **Iteration**: A cycle in the algorithm where weights are updated to approach the optimal value.  
- **K-Nearest Neighbors (KNN)**: A supervised learning algorithm that classifies or predicts based on the nearest neighbors.  
- **K Classes**: The total number of classes in a multi-class classification problem.  
- **K-Means**: A clustering algorithm that divides data into k groups based on similarity.  
- **Kernel**: A function that transforms data, making it separable in high-dimensional spaces.  
- **Keras**: A deep learning library used to build neural networks quickly and easily.  
- **Labeled Data**: A dataset in which each example has an assigned class for training purposes.  
- **Layer Normalization**: A technique that **normalizes the activations** within each layer of the network, **stabilizing and accelerating the training process**. It normalizes the inputs across the features, instead of across the batch like Batch Normalization.
- **Latent space**: is a **representation of data** in a multidimensional space in which similar elements are placed close together. It is created using machine learning, often by reducing the dimensionality of the data to compress it.
- **Learning Rate (α)**: A parameter that controls the speed at which model parameters are updated.  
- **LSTM (Long Short-Term Memory)**: An advanced type of recurrent neural network (RNN) that handles long-term dependencies more effectively by avoiding the vanishing gradient problem. Applications include image generation, automated writing, and the automatic description of images and videos.  
- **Linear Combination (z)**: The weighted sum of inputs and weights, plus a bias:  
    $[ z = (x₁·w₁) + (x₂·w₂) + b ]$  
- **Linear Kernel**: Uses a simple hyperplane to separate classes.  
- **Linear Regression**: A regression algorithm that predicts a continuous value based on a linear relationship between variables.  
- **LLM**: A language model **estimates the probability that a sequence of words will appear**.
- **Log-Loss (Loss Function)**: A loss function used to measure the error in logistic regression.  
- **Logistic Regression**: A classification algorithm that predicts the probability that an observation belongs to a class.  
- **Logit**: The logarithm of the odds ratio, used to model log-linear relationships.  
- **Logit Function**: Transforms any value into a probability between 0 and 1.  
- **loss**: Measures the error on the training data.
- **Majority Voting**: A method used in **One-vs-One** classification where the final class is determined by the most votes among binary classifiers.  
- **Margin**: The distance between the hyperplane and the nearest data points (**support vectors**).  
- **Max-Pooling**: A technique that selects the maximum value within a region of the image.
- **Matrix**: A table of numbers with rows and columns, used for linear transformations.
- **Markov Property**: It implies that the action taken by our agent is conditional solely on the present state and independent of the past states and actions.
- **Mean Squared Error (MSE)**: A loss function that measures error in regression models.  
- **Meta learning** is the art of teaching AI to learn on its own, like a child learning to learn. Meta-learning in machine learning refers to learning algorithms that learn from other learning algorithms.
Most commonly, this means the use of machine learning algorithms that learn how to best combine the predictions from other machine learning algorithms in the field of ensemble learning.
- **Minkowski Distance**: A distance metric that generalizes Euclidean and Manhattan distances. Used in **Clustering, KNN, Geometry**.  
- **Minimum Global (Global Minimum)**: The lowest point of the cost function, representing the smallest possible error.  
- **Minimum Local (Local Minimum)**: A low point in the cost function, which is not necessarily the absolute minimum, where the model may become stuck.  
- **MNIST**: is a dataset of images of handwritten digits, commonly used to train and test machine learning models.
- **Mixture of Experts (MoE)**: is an **architecture** that combines **multiple** specialized **models**, called **“experts,”** to **address complex tasks more efficiently and effectively**.
- **Mixed precision training**: An **optimization tool** that leverages both **16-bit and 32-bit floating-point types** to accelerate model training and reduce memory usage.
- **Multicollinearity**: A phenomenon where two or more features are strongly correlated, negatively affecting the model.
- **Multi-Head Attention Layer**: A mechanism that allows the model to **attend to different parts of the input sequence and learn relationships between them**. It uses multiple "heads" to **focus on different aspects of the data**, improving the model's ability to capture complex patterns.  
- **Multinomial Logistic Regression**: A statistical model that generalizes binary logistic regression for multi-class classification.  
- **Multi-Class Classification**: A problem in which a data point must be assigned to one of **K** available classes.  
- **Neural Network**: A computational model inspired by the human brain, composed of interconnected artificial neurons.  
- **Neuron**: The basic unit of the brain and nervous system, responsible for transmitting information.  
- **Neuron Output (a)**: The final value of a neuron after applying the activation function. 
- **Noise**: In the scattering process, noise is a **random disturbance added to the data** 
- **Non-linearity**: A property that enables a model to learn complex relationships between variables.  
- **Nucleus**: The part of the neuron that contains the cell’s genetic material and processes received information.  
- **Numerical Computation**: Fundamental in machine learning algorithms, which often require **mathematical calculations performed iteratively**, rather than direct analytical solutions.
- **Observations**: The rows in a dataset, each containing information about a single example.  
- **Odds Ratio**: The ratio between the probability of success and the probability of failure.  
- **One-Hot Encoding**: A technique to convert **categorical variables** into numeric form for machine learning models such as logistic regression.  
- **One-vs-All (One-vs-Rest)**: A multi-class classification strategy where a binary classifier is built for each class, distinguishing it from all other classes.  
- **One-vs-One**: A classification strategy in which a binary classifier is trained for each pair of classes, and the final decision is made based on the majority vote.  
- **One-shot learning**: is an object categorization problem, found mostly in computer vision. Whereas most machine learning-based object categorization algorithms require training on hundreds or thousands of examples, one-shot learning aims to classify objects from one, or only a few, examples. 
- **Open System**: is a structure that constantly exchanges matter, energy, or information with its external environment. Unlike isolated systems, these systems have permeable and dynamic boundaries.
- **Outlier Detection**: The process of identifying anomalous data points in a dataset.  
- **Output Layer**: The final layer of a neural network that produces the result.  
- **Overfitting**: When a model is overly complex and fits the training data too closely, leading to poor performance on new data.  
- **Overcoming Catastrophic Forgetting**: Phenomenon in which a **network trained sequentially** on multiple tasks rapidly **forgets** previously acquired knowledge.
- **Parameter**: Are the values ​​that the model learns during the training process
- **Parameter C**: A parameter in **SVM models** that controls the trade-off between a strict separation and a softer margin.  
- **Parameters (θ)**: The model coefficients that are optimized during training.  
- **Persona Bias**: harmful differences in responses generated by the system when assuming different demographic identities.
- **PCA (Principal Component Analysis)**: A traditional algorithm for dimensionality reduction, limited to linear transformations.  
- **Ponderation of Neighbors**: A technique in **KNN classification** that assigns greater weight to nearer neighbors.  
- **Policy**: It is called **the agent’s brain**. It tells us what action to take, given the state.
- **Polynomial Kernel**: Maps data into a more complex space using polynomial functions.  
- **Pooling Layer**: A layer that reduces the dimensions of data (e.g., images) to optimize the network. 
- **Pre-trained models**: are neural networks that have already been trained on large datasets.
- **Probabilistic model**: A model that incorporates randomness and uncertainty, often used to predict distributions or simulate processes that have inherent variability.
- **Q-Function**: Function that estimates the expected value of a certain action in a given state: Q(s, a).
- ***Q-learning**: is a **reinforcement learning algorithm** that teaches an agent which action to perform in each state to **maximize cumulative rewards**.
- **[Prompt Engineering](https://github.com/Mike014/LLM_Concepts/blob/main/Prompt_Engineering.ipynb)**: Creating clear and optimized prompts to guide the model to give correct answers.
- **RBM (Restricted Boltzmann Machine)**: An advanced unsupervised model used to generate missing data, balance datasets, and extract features.  
- **RBF (Radial Basis Function) Kernel**: A kernel that uses a transformation based on the distance between points to separate complex data.  
- **Random noise**: A type of custom augmentation that adds random noise to images, simulating different lighting conditions and sensor noise to make models more robust.
- **Recommendation Systems**: Applications that suggest content based on the clustering of users or products.  
- **Recurrent Neural Networks (RNNs)**: Deep neural networks designed to process sequential data by using previous outputs as inputs for subsequent steps.  
- **ReLU (Rectified Linear Unit)**: The most commonly used activation function, which activates only neurons with positive input.  
- **Reinforcement Learning (RL)**: is based on learning through interaction with the environment. An **agent** (an AI) performs actions in the environment and receives **rewards** (positive or negative) as feedback.
- **Regression**: A statistical technique that estimates the relationship between a continuous dependent variable and one or more independent variables.  
- **Regression Model**: A model that predicts a continuous numerical value, such as concrete strength.  
- **Regression with KNN**: A method that predicts a numerical value by taking the mean or median of the values of the K nearest neighbors.  
- **Reinforcement Learning (RL)**: is a machine learning paradigm in which an agent learns to make decisions by interacting with an environment.
- **Reparameterization Trick**: a **fundamental technique in variational inference**, particularly when working with models such as Variational Autoencoders (VAE).
- **Replay Buffer FIFO**: memory that holds a moving window of the agent’s past experiences.
- **Reverse process**: In diffusion models, the process of removing noise step by step to reconstruct the original data from a noisy sample.
- **Santiago Ramón y Cajal**: The Spanish scientist considered the father of modern neuroscience. 
- **Scalar**: A single number, an isolated value.
- **Self-Attention Mechanism**: Allows the **model to weight words** in the **context of a sentence**. Each word compares itself to all other words in the sentence, calculating how important it is compared to the others (attention weights).
- **Sequential data**: are text, time series, audio.
- **Semantic segmentation**: A deep learning task that involves classifying each pixel in an image into a predefined class. 
- **Sequential Data**: Data organized in a specific order where context from previous elements is crucial.  
- **Sequential Model-API**: A type of model in Keras where layers are stacked sequentially.  
- **Shallow Neural Network**: A neural network with only one or two hidden layers that primarily processes input as vectors.  
- **Sigmoid**: A mathematical function that transforms inputs into a value between 0 and 1.  
- **Synthetic data**: This is **data that is artificially generated by the model**, not from real measurements or observations.
- **Soft Margin**: A margin that allows some misclassifications to improve the model's generalization.  
- **SoftMax Probability**: The probability assigned to each class in a **SoftMax model**, computed by transforming the dot products of data and model parameters.  
- **SoftMax Regression**: A variant of logistic regression that assigns probabilities to multiple classes by transforming outputs into a probability distribution.  
- **Softmax**: Converts the output of a layer into probabilities for class membership.  
- **Stride**: The step size with which the convolutional filter moves across the image.  
- **Soma**: The main body of a neuron that contains its nucleus.  
- **Standardization of Features**: The process of scaling features to make them comparable and reduce unbalanced impact on predictions.  
- **Stochastic Gradient Descent (SGD)**: A variant of gradient descent that updates weights using **one sample** at a time.  
- **Supervised Learning**: A learning method in which the model is trained on labeled data.  
- **Support Vector Machines (SVM)**: A supervised machine learning algorithm used for classification and regression.  
- **Support Vector Regression (SVR)**: A variant of SVM used for predicting continuous values.  
- **Support Vectors**: Data points that are closest to the hyperplane and influence class separation.  
- **Target**: The dependent or output variable that the model is intended to predict.  
- **Target Network**: A stable and less frequently updated copy of the main Q-network, used to compute training targets.
- **Temporal Context**: Relevant information over time that influences the processing of sequential data.  
- **Tensors**: Fundamental data structure in artificial intelligence, providing a means of storing both input and output data within a model,  A multi-dimensional array, a generalization of vectors and matrices.
- **Theta Coefficient**: Values that indicate **how much each feature affects the prediction**.  
- **Threshold (Decision Threshold)**: The value (e.g., **0.5**) beyond which an observation is assigned to a class.  
- **Training Set**: The dataset used to train the model.  
- **Transfer Learning**: It is a technique that allows you to reuse pre-trained models on large datasets for new and related tasks.
- **Transpose convolution**: An operation that reverses the effects of convolution, often used for up-sampling in image processing.
- **Transpose**: Swaps rows and columns of a matrix.
- **t-distributed stochastic neighbor embedding (t-SNE)**:	A dimensionality reduction technique used for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.
- **Underfitting**: When the model is too simple to capture the underlying patterns in the data, leading to inaccurate predictions.  
- **Unsupervised Learning**: Machine learning without labels, where the model finds patterns in the data.  
- **UpSampling2D (Decoding for Autoencoders)**: The inverse operation of pooling, which increases the input size by replicating its values.
- **Variational Autoencoders (VAE)**: Generative machine learning models that **create new variations of input data**. They are **autoencoders that encode latent variables** in a **probabilistic and continuous**, not discrete, way. They use **variational inference** to generate new data samples.
- **val_loss**: Measures the error on the test/validation data.
- **Values of K**: The number of neighbors considered when determining the class or target value in **KNN**.  
- **Vanishing Gradient Problem**: A problem where gradients become too small during training, making the learning process slow and ineffective.  
- **Variance**: A measure of how much the model’s predictions fluctuate when trained on different subsets of the dataset. High variance often leads to **overfitting**.  
- **VGG16**: A convolutional neural network model pre-trained on the ImageNet data set, commonly used in transfer learning for tasks involving image classification.
- **Vector**: An ordered list of numbers, representing a point in space.
- **Weight (w)**: A numerical value that determines the importance of an input in a neuron.  
- **Word Embedding**: is a collective term in natural language processing for a set of modeling techniques in which words or phrases from a vocabulary are mapped into vectors of real numbers.
- **Weight Update**: The process of updating a weight using the formula:  
    ${\text{new}} = w_{\text{old}} - \alpha \cdot \text{gradient}$
- **Zero-Sum Game**:	A situation in competitive contexts where gain or loss of participants is exactly balanced by the losses or gains of another participant.

# Linear Algebra Glossary for Deep Learning

## Fundamental Objects
- **Scalar**: A single real number, denoted as $a \in \mathbb{R}$. It represents a quantity that has only magnitude but no direction.
- **Vector**: An ordered one-dimensional array of numbers, denoted as $x \in \mathbb{R}^n$. It can be geometrically interpreted as an arrow in space with both magnitude and direction.
- **Matrix**: A two-dimensional array of numbers, denoted as $A \in \mathbb{R}^{m \times n}$, where $m$ is the number of rows and $n$ is the number of columns. It represents a linear transformation between vector spaces.
- **Tensor**: A generalization of vectors and matrices to higher dimensions, denoted as $T \in \mathbb{R}^{d_1 \times \dots \times d_n}$. It is a fundamental data structure in deep learning for representing multidimensional information.

## Fundamental Operations
- **Transpose**: The operation $A^T$ that swaps the rows and columns of a matrix. If $A$ is an $m \times n$ matrix, then $A^T$ is an $n \times m$ matrix.
- **Matrix Multiplication**: The operation $C = AB$ defined only if the number of columns in $A$ equals the number of rows in $B$. It represents the composition of linear transformations.
- **Hadamard Product**: The operation $A \circ B$ that multiplies two matrices of the same dimensions element by element. Commonly used in neural networks for masks and gating mechanisms.
- **Dot Product**: The operation $x^T y = \sum x_i y_i$ between two vectors. It measures the projection of one vector onto another and can be interpreted as an indicator of alignment between vectors.
- **L1 Norm**: The sum of the absolute values of a vector's elements, $\|x\|_1 = \sum |x_i|$. Also known as the Manhattan distance.
- **L2 Norm**: The square root of the sum of squares of a vector's elements, $\|x\|_2 = \sqrt{\sum x_i^2}$. Corresponds to the Euclidean distance.
- **Infinity Norm**: The maximum absolute value among a vector's elements, $\|x\|_\infty = \max |x_i|$.
- **Frobenius Norm**: The matrix analog of the L2 norm, $\|A\|_F = \sqrt{\sum A_{ij}^2}$. Measures the overall magnitude of a matrix.

## Core Concepts
- **Linear Dependence**: A vector is linearly dependent if it can be written as a linear combination of other vectors. A set of vectors is linearly dependent if at least one vector can be expressed as a linear combination of the others.
- **Span**: The set of all possible linear combinations of a set of vectors. It represents the vector space generated by these vectors.
- **Column Space**: The span of a matrix's columns. The equation $Ax = b$ has solutions if and only if $b$ belongs to the column space of $A$.
- **Invertibility**: A matrix is invertible only if it is square and has linearly independent columns. The inverse $A^{-1}$ satisfies $A A^{-1} = A^{-1} A = I$.
- **Singularity**: A matrix is singular if at least one of its eigenvalues is zero. Singular matrices are not invertible.

## Matrix Decompositions
- **Eigendecomposition**: The factorization $A = V \Lambda V^{-1}$ where $\Lambda$ is a diagonal matrix containing the eigenvalues and $V$ is a matrix whose columns are the eigenvectors. It allows analysis of the scaling and stretching directions of a linear transformation.
- **Symmetric Decomposition**: A simplified form of eigendecomposition for symmetric matrices, $A = Q \Lambda Q^T$, where the eigenvectors (columns of $Q$) are orthogonal to each other.
- **Singular Value Decomposition (SVD)**: The universal factorization $A = U D V^T$ that works for all real matrices. $U$ and $V$ are orthogonal matrices, and $D$ is a diagonal matrix containing the singular values.
- **Principal Component Analysis (PCA)**: Derived from SVD applied to centered data. Used for dimensionality reduction while preserving data variance.
- **Moore-Penrose Pseudoinverse**: The generalization $A^+ = V D^+ U^T$ of the inverse for non-square or singular matrices.

## Special Matrices
- **Diagonal Matrix**: A matrix where $A_{ij} = 0$ if $i \neq j$. Allows for fast multiplication and inversion.
- **Symmetric Matrix**: A matrix where $A = A^T$. Has real eigenvalues.
- **Orthogonal Matrix**: A matrix that satisfies $A^T A = I$. Has the property $A^{-1} = A^T$ and preserves lengths and angles in transformations.
- **Positive Definite Matrix**: A matrix $A$ for which $x^T A x > 0$ for every $x \neq 0$. All of its eigenvalues are strictly positive.
- **Positive Semidefinite Matrix**: A matrix $A$ for which $x^T A x \geq 0$ for every $x$. All of its eigenvalues are greater than or equal to zero.

## Eigendecomposition and Optimization
- **Eigenvalues and Eigenvectors**: An eigenvector $v$ of a matrix $A$ is a non-zero vector such that $A v = \lambda v$, where $\lambda$ is the corresponding eigenvalue.
- **Optimization Form**: Maximize $x^T A x$ subject to $\|x\| = 1$. The solution is the eigenvector corresponding to the largest eigenvalue.
- **Eigenvalue Implications**: If any eigenvalue $\lambda = 0$, then the matrix $A$ is singular.

## Singular Value Decomposition (SVD)
- **General Form**: $A = U D V^T$ where:
  - $U$ contains the left singular vectors (orthogonal)
  - $V$ contains the right singular vectors (orthogonal)
  - $D$ is a diagonal matrix (possibly rectangular) of singular values
- **Relationship with Eigenvalues**: The singular values are $\sqrt{\lambda}$ where $\lambda$ are the eigenvalues of the matrices $A^T A$ or $AA^T$.

## Moore-Penrose Pseudoinverse
- **Wide Matrix ($m < n$)**: Infinite solutions exist; the pseudoinverse provides the solution with minimum norm.
- **Tall Matrix ($m > n$)**: An exact solution might not exist; the pseudoinverse minimizes $\|Ax - y\|_2$.
- **Formula (SVD-based)**: $A^+ = V D^+ U^T$ where $D^+$ inverts the non-zero singular values.
- **Usage**: Robust pseudo-inversion for non-invertible or non-square matrices.

## Principal Component Analysis (PCA)
- **Objective**: Project data onto a lower-dimensional subspace with minimal reconstruction error.
- **Projection Formula**: $r(x) = D D^T x$.
- **Optimization Objective**: Minimize $\|X - X D D^T\|_F^2$, subject to $D^T D = I$.
- **Case $l = 1$**: Maximize $d^T X^T X d$, with $\|d\| = 1$.
- **Solution**: The first eigenvector of $X^T X$ or the first right singular vector of the SVD.
- **With SVD**: The principal components are the first columns of $V$ in the decomposition $X = U D V^T$.

# **Linear Algebra**

| Mathematical Concept | Mathematical Description |
|----------------------|-------------------------|
| **Vector Space** | Set of all vectors that can be generated through linear combinations |
| **Orthonormal Basis** | Minimum set of independent vectors that generate the entire space | 
| **Span** | Everything that can be generated from a basis | 
| **Linear Dependence** | A vector is redundant, can be obtained from others | 
| **Linear Transformation (Ax = b)** | Application of a matrix to a vector |
| **Eigenvectors / Eigenvalues** | Invariant directions under transformation, and their "scaling" coefficient | 
| **Orthogonality** | Two vectors do not interfere (90° angle) |
| **Orthogonal Matrix** | Preserves norms, angles, spatial identity | 
| **Norm (L2)** | Measure of distance or intensity of a vector | 
| **SVD (A = UDVᵀ)** | Universal decomposition of a transformation | 
| **Moore-Penrose Pseudoinverse** | Recovery of a solution even when A is not invertible | 
| **PCA** | Projection on directions that explain more variance | 
| **Symmetric Matrix** | A = Aᵀ: perfectly balanced | 
| **Singularity** | Non-invertible matrix → loss of information | 
| **Vector Projection (x → d dᵀ x)** | Reducing reality to a single dimension | 
| **Frobenius Norm** | Measure of total energy of a transformation | 

# Probability & Information Theory – Summary Table

| **Section**              | **Key Concept**                                                                 | **Formula / Important Notes**                                                                 |
|--------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Why probability in AI    | Managing uncertainty, guiding algorithms and analysis                           | Two main uses: design and analysis                                                          |
| Types of probability      | Frequentist vs Bayesian                                                        | Frequentist = repeated events<br>Bayesian = degree of confidence                            |
| Random variables          | Variables that take values ​​according to a distribution                       | Discrete (PMF) / Continuous (PDF)                                                          |
| PMF – Discrete            | Exact probability of each state                                                | $$ \sum_x P(x) = 1 $$                                                                       |
| PDF – Continuous          | Probability density (not point value)                                          | $$ \int p(x) dx = 1 $$, $$ P(x = c) = 0 $$                                                  |
| Marginalization           | Isolate one variable, ignoring the others                                      | Discrete: $$ P(x) = \sum_y P(x, y) $$ <br> Continuous: $$ p(x) = \int p(x, y) dy $$         |
| Conditionalization        | Probability of `y` given that `x` has happened                                 | $$ P(y | x) = \frac{P(x, y)}{P(x)} $$                                                      |
| Independence              | No mutual influence between `x` and `y`                                        | $$ P(x, y) = P(x)P(y) $$                                                                    |
| Conditional independence  | `x` and `y` independent given `z`                                              | $$ P(x, y | z) = P(x | z)P(y | z) $$                                                       |
| Expectation               | Weighted mean of a function on one variable                                    | Discrete: $$ \mathbb{E}[f(x)] = \sum_x P(x)f(x) $$ <br> Continuous: $$ \int p(x)f(x) dx $$  |
| Variance                  | Dispersion about the mean                                                      | $$ \text{Var}(f(x)) = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2] $$                           |
| Covariance                | How two variables vary together                                                | $$ \text{Cov}(x, y) = \mathbb{E}[(x - \mathbb{E}[x])(y - \mathbb{E}[y])] $$                |
| Covariance ≠ independence | Cov = 0 does not imply independence                                            | There can be non-linear dependence                                                         |
| Covariance matrix         | Relations and variance for vectors of variables                                | $$ \text{Cov}(x)_{i,j} = \text{Cov}(x_i, x_j) $$                                           |
| Probabilistic models      | Factorization of the distribution into simpler factors                         | Reduces parameters, useful for AI                                                          |
| Direct (Bayesian) models  | Graphs with directed edges, explicit conditioning                              | $$ P(x) = \prod_i P(x_i \mid \text{parents}(x_i)) $$                                        |
| Indirect (Markov) models  | Undirected graphs, positive but unnormalized factors (φ)                       | $$ P(x) = \frac{1}{Z} \prod_i \phi(C_i) $$, with $$ Z $$ normalization constant             |
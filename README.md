# PyTorch Deep Learning Projects 🧠🔥

![Model Performance](model_performance.png)
*(Confusion Matrix from the Handwritten Digits Classification Project)*

## 📌 Repository Overview
This repository contains a collection of three distinct deep learning projects built using **PyTorch**. These projects demonstrate various core concepts of machine learning, ranging from regression analysis and data loading pipelines to building full Multilayer Perceptron (MLP) classifiers.

## 📂 Project List

### 1. Employee Bonus Predictor 💰
* **File:** `bonus_prediction_nn.ipynb`
* **Type:** Regression (Neural Network)
* **Description:** A linear regression model implemented as a neural network to predict fair employee bonuses.
* **Key Tech:** Analyzes features like *Performance, Experience,* and *Projects Completed* to output a continuous financial value. Uses MSE Loss and SGD/Adam optimizers.
* **Dataset:** `bonus.csv` (Included).

### 2. Handwritten Digits Classification 🔢
* **File:** `digits_classification.ipynb`
* **Type:** Multi-Class Classification (MLP)
* **Description:** A Feed-Forward Neural Network (Multilayer Perceptron) trained on the famous **MNIST** dataset to recognize handwritten digits (0-9).
* **Key Tech:** * Achieved **97%+ Accuracy**.
    * Uses `CrossEntropyLoss` and backpropagation.
    * Includes a confusion matrix visualization (shown above) to analyze misclassifications.

### 3. FashionMNIST Data Pipeline 👕
* **File:** `datasets_dataloader.ipynb`
* **Type:** Data Engineering / Computer Vision
* **Description:** A deep dive into PyTorch's `Dataset` and `DataLoader` primitives. It demonstrates how to download, transform, and visualize batch data using the **FashionMNIST** dataset.
* **Key Tech:** Image transformation (`ToTensor`), batch processing, and grid visualization of training data.

## 🛠️ Technologies & Libraries
* **PyTorch** (Core Deep Learning Framework)
* **Torchvision** (Datasets & Transforms)
* **Pandas** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)
* **Scikit-Learn** (Metrics)

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/PyTorch-Deep-Learning-Projects.git](https://github.com/yourusername/PyTorch-Deep-Learning-Projects.git)
    ```
2.  Install dependencies:
    ```bash
    pip install torch torchvision pandas matplotlib seaborn scikit-learn
    ```
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4.  Open any of the `.ipynb` files to run the specific project.

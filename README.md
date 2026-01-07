# 🧠 PyTorch Deep Learning Projects

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

![Model Performance](model_performance.png)
*> Confusion Matrix visualization from the Handwritten Digits Classification Project*

## 📌 Repository Overview

This repository contains a collection of foundational deep learning projects built using the **PyTorch** framework.

These projects are designed to serve as a practical introduction to the building blocks of AI. They bridge the gap between traditional machine learning and deep learning by covering essential workflows: starting from manual data handling and linear regression implemented as a neural network, moving to data engineering pipelines, and finally building full **Multilayer Perceptrons (MLPs)** for computer vision tasks.

---

## 📂 Project Roadmap

| Project | Domain | Key Concept | Notebook |
| :--- | :--- | :--- | :--- |
| **1. Bonus Predictor** 💰 | Regression | Linear Regression via Neural Network | `bonus_prediction_nn.ipynb` |
| **2. Digit Classifier** 🔢 | Computer Vision | Multilayer Perceptron (MLP) | `digits_classification.ipynb` |
| **3. Fashion Pipeline** 👕 | Data Engineering | Datasets & DataLoaders | `datasets_dataloader.ipynb` |

---

## 1. Employee Bonus Predictor 💰

* **Type:** Regression / Single-Layer Neural Network
* **Dataset:** `bonus.csv` (Custom dataset included in repo)

**The Challenge:** Companies need a fair, mathematical way to determine employee bonuses based on merit.

**The Logic:**
This project explores how **Linear Regression**—the cornerstone of statistical learning—can be implemented using the architecture of a Neural Network. Instead of using high-level tools like Scikit-Learn, we build a PyTorch model from scratch. The model learns weights for three features to predict a continuous financial value:
1.  Performance Rating (1-10)
2.  Years of Experience
3.  Projects Completed

**Key Learning Outcomes:**
* Converting raw Pandas DataFrames into floating-point PyTorch Tensors.
* Understanding the **Mean Squared Error (MSE)** loss function.
* Implementing the full training loop manually:
    $$\text{Forward Pass} \rightarrow \text{Calculate Loss} \rightarrow \text{Backpropagation} \rightarrow \text{Optimizer (SGD)}$$

---

## 2. Handwritten Digits Classification 🔢

* **Type:** Multi-Class Classification / Multilayer Perceptron (MLP)
* **Dataset:** MNIST (Built-in via Torchvision)

**The Challenge:** Computers see images as grids of numbers. We need a model that can recognize patterns in those grids to identify the number written (0-9).

**The Logic:**
Known as the "Hello World" of Deep Learning, this project builds a Feed-Forward Neural Network. It takes a 28x28 pixel image, "flattens" it into a massive 1D vector of 784 pixels, and passes it through hidden layers.
* **Input Layer:** 784 nodes.
* **Hidden Layers:** Linear layers activated by **ReLU** to introduce non-linearity.
* **Output Layer:** 10 nodes (representing probability scores for digits 0-9).

**Key Learning Outcomes:**
* Using `CrossEntropyLoss` for multi-class classification.
* Evaluation metrics: Calculating accuracy and visualizing mistakes using a Confusion Matrix.
* Saving (`torch.save`) and loading trained models for future inference.

---

## 3. FashionMNIST Data Pipeline 👕

* **Type:** Data Engineering / Computer Vision Basics
* **Dataset:** FashionMNIST (Built-in via Torchvision)

**The Challenge:** Real-world datasets are massive. You cannot load millions of images into RAM at once.

**The Logic:**
Deep learning requires processing data in small "mini-batches." This notebook focuses exclusively on PyTorch's data primitives: `Dataset` and `DataLoader`. It uses FashionMNIST (clothing items) to demonstrate efficient data handling.

**Key Learning Outcomes:**
* **Transforms:** Using `torchvision.transforms.ToTensor` to convert raw images into normalized tensors suitable for GPUs.
* **DataLoaders:** Configuring batch sizes (e.g., 64 images) and shuffling logic to ensure the model doesn't memorize the order of data.
* **Visualization:** Iterating through a generated batch to display a grid of images with their corresponding class labels, ensuring data integrity before training begins.

---

## 🛠️ Technologies & Libraries

* **PyTorch:** Core framework for defining tensors, automatic differentiation, and neural network graphs.
* **Torchvision:** Used for downloading standard datasets (MNIST/FashionMNIST) and applying image transformations.
* **Pandas:** Used for handling structured CSV data (in the Bonus Predictor).
* **Matplotlib & Seaborn:** Used for visualizing training loss curves, sample data grids, and confusion matrices.
* **Scikit-Learn:** Used for train/test splitting and generating classification metrics.

---

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/PyTorch-Deep-Learning-Projects.git](https://github.com/yourusername/PyTorch-Deep-Learning-Projects.git)
    cd PyTorch-Deep-Learning-Projects
    ```

2.  **Install dependencies**
    ```bash
    pip install torch torchvision pandas matplotlib seaborn scikit-learn
    ```

3.  **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```

4.  **Run the projects**
    Click on any of the `.ipynb` files in the dashboard to open them and run the code cells.

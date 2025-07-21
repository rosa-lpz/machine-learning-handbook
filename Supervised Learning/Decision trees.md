# Decision Trees

**Decision Trees** are a type of **supervised machine learning algorithm** used for both **classification** and **regression** tasks. They are easy to understand and interpret, making them popular for a variety of predictive modeling problems.

------

### **Basic Concepts**

1. **Structure**:
   - **Root Node**: The top node that represents the entire dataset, which gets split.
   - **Decision Nodes**: Nodes that split the data into further subsets.
   - **Leaf Nodes (Terminal Nodes)**: Nodes that represent the final output or prediction.
2. **Splitting**:
   - Data is split based on features to create branches. Each split aims to increase the **purity** of the resulting subsets.
3. **Criteria for Splitting**:
   - **Classification**:
     - **Gini Impurity**
     - **Entropy (Information Gain)**
   - **Regression**:
     - **Mean Squared Error (MSE)**
     - **Mean Absolute Error (MAE)**

------

### **Example of Classification Tree Split**

Suppose we want to classify if someone will buy a product based on **Age** and **Income**:

```
                [Age < 30?]
                /        \
             Yes         No
          [Income > 50k?]   Buy = Yes
           /       \
         No        Yes
      Buy = No   Buy = Yes
```

------



###  **Advantages**

- Easy to understand and interpret.
- Requires little data preprocessing.
- Handles both numerical and categorical data.
- Can model non-linear relationships.

------

### **Disadvantages**

- Prone to **overfitting**.
- Small changes in data can lead to very different trees (high variance).
- Biased towards features with more levels.

------



### **Improvements and Variants**

- **Pruning**: Reduces overfitting by trimming branches.
- **Random Forests**: Combines multiple decision trees for better accuracy and robustness.
- **Gradient Boosted Trees (e.g., XGBoost, LightGBM)**: Build trees sequentially to reduce errors.


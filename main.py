import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

X = pd.read_csv('logisticX.csv')
y = pd.read_csv('logisticY.csv')

X_data = X.values
y_data = y.values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost_history = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X @ theta)
        cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta = np.zeros(n + 1)
        X_with_intercept = np.column_stack((np.ones(m), X))
        
        for i in range(self.iterations):
            h = self.sigmoid(X_with_intercept @ self.theta)
            gradient = (1/m) * X_with_intercept.T @ (h - y)
            self.theta -= self.learning_rate * gradient
            cost = self.cost_function(X_with_intercept, y, self.theta)
            self.cost_history.append(cost)
        
        return self
    
    def predict_proba(self, X):
        m = X.shape[0]
        X_with_intercept = np.column_stack((np.ones(m), X))
        return self.sigmoid(X_with_intercept @ self.theta)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Question 1: Logistic Regression
custom_lr = CustomLogisticRegression(learning_rate=0.1)
custom_lr.fit(X_scaled, y_data)
print(f"Final Cost Function Value : {custom_lr.cost_history[-1]:.6f}")
print("Learned Coefficients:", custom_lr.theta)

# Question 2: Cost Function vs Iterations Plot
plt.figure(figsize=(10, 6))
plt.plot(custom_lr.cost_history[:50])
plt.title('Cost Function vs Iterations (First 50 Iterations)')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.tight_layout()
plt.grid()
plt.savefig('cost_function_plot.png')
plt.close()

# Question 3: Data Points and Decision Boundary
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='viridis')
plt.colorbar(scatter)

x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))
Z = custom_lr.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
plt.title('Data Points with Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.savefig('decision_boundary.png')
plt.close()

# Question 4: Adding Squared Features
X_squared = np.column_stack((X_data, X_data[:, 0]**2, X_data[:, 1]**2))
X_squared_scaled = StandardScaler().fit_transform(X_squared)

custom_lr_squared = CustomLogisticRegression(learning_rate=0.1)
custom_lr_squared.fit(X_squared_scaled, y_data)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_squared[:, 2], X_squared[:, 3], c=y_data, cmap='viridis')
plt.colorbar(scatter)

x_min, x_max = X_squared[:, 2].min() - 1, X_squared[:, 2].max() + 1
y_min, y_max = X_squared[:, 3].min() - 1, X_squared[:, 3].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))
Z = custom_lr_squared.predict(StandardScaler().fit_transform(np.column_stack((np.zeros(10000), np.zeros(10000), xx.ravel(), yy.ravel()))))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
plt.title('Squared Features: Data Points with Decision Boundary')
plt.xlabel('Squared Feature 1')
plt.ylabel('Squared Feature 2')
plt.tight_layout()
plt.savefig('squared_features_decision_boundary.png')
plt.close()

# Question 5: Confusion Matrix and Metrics
y_pred = custom_lr.predict(X_scaled)
cm = confusion_matrix(y_data, y_pred)
print("Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("\nMetrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
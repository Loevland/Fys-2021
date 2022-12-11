# Classifiers
---
## Linear Regression
### Assumptions
- Linear relationship between independent variables(X) and dependent variables(mean) of y  
- Multivariate normality (X must be normal or converted to be more normal)
- No or little multicollinearity (Independent variables should not be too correlated)
- No auto-correlation (All input are independent)
- Homoscedasticity (Same variance)

### Basic Ideas
- Regression = Predict continous values
- Linear function to predict the values
- Gradient descent based upon cost function gradient to find good approximation

#### Approxmiation line
$y = wx + b$

#### Cost function
$$MSE = \frac{\sum(r - (y))^2}{N}$$

#### Gradient descent Update Rules
w = w - a * dw <br/>
b = b - a * db

### Pseudocode
```python
def gradient_descent():
  for _ in iterations:
    y_pred = dot(X, weight) + bias
    dw = 1/n * dot(X, (y_pred - y))
    db = 1/n * sum(y_pred - y)
    self.w -= lr * dw
    self.b -= lr * db

def predict():
  y_pred = dot(X, weight) + bias
```
---

## Naive Bayes
### Assumptions
- Input is independent of each other
- Classes follow a distribution

### Basic Ideas
- Find probability for input belonging to each class for their probability function, with prior as well, if defined.
- Pick class with the highest probability of all

### Pseudocode
```python
def predict(x):
  for each class:
    y_pred = prior * _pdf(x)

  pred = max(y_preds)

def _pdf():
  gaussian or gamma probability
```
---
## Logistic Discrimination
### Assumptions
- Data is independent
- No or little multicollinearity
- Linearity of independent variables?
- Large sample size?

### Basic Ideas
- Train with e.g. gradient descent for weights
- Make predictions with weights
- Weights are trained to find the optimal line(s) to separate the data
- Weights maximizes the distance to the decision boundary $$wx + b$$
- Weights make it so probabilities are maximized towards a class for each datapoint

### Pseudocode
```python
def fit():
  for _ in iterations:
    lin_model = dot(X, w)
    y_pred = pdf(lin_model)
    delta_w = dot((y - y_pred), X)
    w += lr * delta_w

def predict():
  pred = dot(X, w)
```

---
## Perceptron
### Assumptions
- Binary classification
- Data is linearly separable

### Basic Ideas
- Train weights
- Make prediction with an activation function (e.g. sigmoid or relu, and weights + bias)

#### Update weights
$$w = w + \delta w$$
$$\delta w = lr * (y - y_pred) * x$$

Weights are pushed towards positive or negative target class when a misclassification occurs

### Pseudocode
Check own implementation

---
## Multilayer-Perceptron
### Assumptions
- Any?

### Basic Ideas
- Train the weights of the MLP and make predictions
- Weights are updated using backpropagation
- Weight update factor is determined by derivation of loss functions

### Pseudocode
```python
init_random_weights()
hl, ol = Layers()

for _ in iterations:
  hl_pred = hl.predict(X)
  ol_pred = ol.predict(hl_pred)
  loss = calculate_loss(y, ol_pred)

  delta_w = loss * pred_func_der(ol_pred)
  ol_update = lr * dot(delta_w, hl_pred)

  hl_error = dot(ol.weights(), delta_w)
  hl_err_update = pred_func_der(hl_pred) * hl_error
  hl_update = lr * dot(hl_err_update, hl_pred)

```
---
## K-means Clustering
### Assumptions
- Any?

### Basic Ideas
- Make clusters by labeling similar to their own class
- Iterative approach finding similarities in data

### Pseudocode
```python
init_random_centroids()
for _ in iterations:
  assign_to_clusters(X)
  old_centroids = centroids
  centroids = calculate_new_centroids()

  if old_centroids == centroids:
    break

def assign_to_clusters(x):
  centroid = shortest_distance(x, centroids)
```

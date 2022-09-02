# Bayesian Decision Theory

## Formulas
$$ P(C_i|X)= { P(C_i)P(X|C_i) \over P(X) } $$
$$ P(X) = \sum_{i=1}^{K} P(X|C_i)P(C_i) $$
$$ P(C_i | X) = { P(C_i)P(X|C_i) \over \sum_{i=1}^{K} P(X|C_i)P(C_i) } $$
$$ \sum_{i=1}^{K} P(C_i) = 1 $$
$$ \sum_{i=1}^{K} P(C_i|X) = 1 $$


### Binary Classification with 2 classes

$$ P(C_1|X) + P(C_2|X) = 1 $$
$$ P(C_1|X) = 1 - P(C_2|X) $$
$$ P(C_2|X) = 1 - P(C_1|X) $$

### Determine output class
- Determine the output class based on what class has the highest probability $max(P(C_1|X), P(C_2|X))$
- The same applies if there are more than 2 classes
- Choosing the class with the highest probability minimizes the risk of loss, as P(Wrong) = 1 - P(Correct)

$C_1$ if $P(C_1|X) > P(C_2|X)$

$C_2$ if $P(C_2|X) > P(C_1|X)$

## Loss and Risk
- $C_k$ = Correct Class
- $a_i$ = The Predicted Class

A classification is correct $C_k = a_i$ when $k = i$

A wrong prediction is called a **loss** and is written as
$$ loss = \lambda_{ik} $$
where *i* is index of the predicted class and *k* is the index of the correct class

Risk can be calculated using the formula
$$ R(a_i|X) = \sum_{k=1}^K \lambda_{ik}P(C_k|X) $$

#### 0/1 Loss
This is when the loss can only be 0 or 1.
When the loss can be only 0 or 1 the $\lambda$ can be removed, giving the formula
$$ R(a_i|X) = \sum_{k=1, k \ne i}^K P(C_k|X) $$

The risk of a 0/1 loss can also be calculated using the following formula
$$ R(a_i|X) = 1 - P(C_i|X) $$ 

The risk is minimized when the class with the maximum probability is selected

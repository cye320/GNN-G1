import numpy as np

# 1.2 Matrix generation function
def SampleGenerator(m, n):
    return np.random.binomial(1, 1 / m, (m, n))

# 1.3 Input generation function
def InputsGenerator(n):
    return np.random.normal(np.zeros(n), np.sqrt(0.5 / n) * np.ones(n))

# 1.3 Linear model sample generator
def TrainingSetGenerator(A, q):
    m, n = A.shape
    X = InputsGenerator(n)
    Y = A @ X + InputsGenerator(m)

    for i in range(q - 1):
        X = np.column_stack((X, InputsGenerator(n)))
        Y = np.column_stack((Y, A @ X[:, -1] + InputsGenerator(m)))

    return X.T, Y.T

# Parameter settings
m = 10**2
n = 10**2
Q = 10**3

# Generate matrix A
A = SampleGenerator(m, n)

# Generate training set and test set
T_x, T_y = TrainingSetGenerator(A, Q)
T_x_prime, T_y_prime = TrainingSetGenerator(A, Q)

# Least squares solution for H
H = T_y.T @ T_x @ np.linalg.inv(T_x.T @ T_x)

# Calculate mean squared error (MSE)
def mse_loss(H, x, y):
    y_pred = np.dot(x, H.T)
    return np.mean(np.sum((y - y_pred) ** 2, axis=1))

# Calculate MSE on training and test sets
train_mse = mse_loss(H, T_x, T_y)
test_mse = mse_loss(H, T_x_prime, T_y_prime)

# Output results
print(f"The MSE on the training set is: {train_mse}")
print(f"The MSE on the test set is: {test_mse}")


# 3.2
# 1.4 Sign model sample generator
def TrainingSetGeneratorSign(A, q):
    m, n = A.shape
    X = InputsGenerator(n)
    Y = A @ X + InputsGenerator(m)

    for i in range(q - 1):
        X = np.column_stack((X, InputsGenerator(n)))
        Y = np.column_stack((Y, A @ X[:, -1] + InputsGenerator(m)))

    return X.T, np.sign(Y.T)

# Generate training and test sets for sign model
T_x_sign, T_y_sign = TrainingSetGeneratorSign(A, Q)
T_x_prime_sign, T_y_prime_sign = TrainingSetGeneratorSign(A, Q)

# Least squares solution for H
H_sign = T_y_sign.T @ T_x_sign @ np.linalg.inv(T_x_sign.T @ T_x_sign)

# Calculate MSE on training and test sets for sign model
train_mse_sign = mse_loss(H_sign, T_x_sign, T_y_sign)
test_mse_sign = mse_loss(H_sign, T_x_prime_sign, T_y_prime_sign)

# Output results
print(f"The MSE on the training set for the sign model is: {train_mse_sign}")
print(f"The MSE on the test set for the sign model is: {test_mse_sign}")

# 3.3
# Use larger dimensions for n and m
n_large = 100
m_large = 100
Q = 100
# Generate matrix A and samples for large dimensions
A_large = SampleGenerator(m_large, n_large)
T_x_large, T_y_large = TrainingSetGenerator(A_large, Q)
T_x_prime_large, T_y_prime_large = TrainingSetGenerator(A_large, Q)

# Least squares solution for H
H_large = T_y_large.T @ T_x_large @ np.linalg.inv(T_x_large.T @ T_x_large)

# Calculate MSE on training and test sets for large dimensions
train_mse_large = mse_loss(H_large, T_x_large, T_y_large)
test_mse_large = mse_loss(H_large, T_x_prime_large, T_y_prime_large)

# Output results
print(f"The MSE on the training set for the large dimension model is: {train_mse_large}")
print(f"The MSE on the test set for the large dimension model is: {test_mse_large}")

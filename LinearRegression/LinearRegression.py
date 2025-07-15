
import numpy as np
import matplotlib.pyplot as plt
def subgradientDescent(x, y, alpha ,iteration=100000, epsilon = 3e-2):
    rows , columns = x.shape
    a = np.zeros(columns)
    b = 0.0
    theta = np.concatenate([a,[b]])
    for i in range(1, iteration+1):
        residuals = x.dot(a) + b  - y
        loss = residuals**2 # squared loss
        mstar = np.argmax(loss)
        gradient = 2 * residuals[mstar] * np.append(x[mstar], 1)
        new_alpha = alpha / np.sqrt(i) #
        theta = theta - new_alpha * gradient
        a = theta[:-1]
        b = theta[-1]
        if np.linalg.norm(new_alpha * gradient )< epsilon:
            print(f"Converged at Iteration: {i}")

            break
    return np.concatenate([a,[b]])

def generateData():
    np.random.seed(0)
    X = np.linspace(0, 10, 10)
    y = 10 * X + 5 + np.random.randn(10) * 5  # y = 10x + 5 + noise
    X = X.reshape(-1, 1)
    return X, y
def leastSquaresSolution(X, y):
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    # solution
    theta, res, rank , s = np.linalg.lstsq(X, y,rcond=None)
    a = theta[0]
    b = theta[1]
    return a, b


if __name__ == '__main__':
    # Modified Loss Function Regression
    X, y = generateData() # generate data
    theta_subgrad = subgradientDescent(X, y, alpha=0.1)
    a_subgrad = theta_subgrad[0]
    b_subgrad = theta_subgrad[1]
    a_leastsquare,b_leastsquare = leastSquaresSolution(X, y)
    print(f"Subgradient Descent Solution: a = {a_subgrad}, b = {b_subgrad}")
    print(f"Least Squares Solution: {a_leastsquare}, b = {b_leastsquare}")

    # Plotting the results
    X_plot = np.linspace(0, 10, 100)
    y_leastsqaure = a_leastsquare * X_plot + b_leastsquare
    y_subgrad = a_subgrad * X_plot + b_subgrad

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X_plot, y_leastsqaure, color='orange', label='Least Squares Regression')
    plt.plot(X_plot, y_subgrad, color='blue', linestyle='--', label='Modified Loss Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Comparison of Regression Lines')
    plt.legend()
    plt.grid(True)
    plt.show()


















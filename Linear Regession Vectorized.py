import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
np.random.seed(15)

df = pd.read_csv(r'C:\Users\biswa\OneDrive\Desktop\spectacle_price_dataset.csv')#Write Dataset Location
x = df.values
p = df['Price ($)'].values
m_train = len(p)
no_of_iterations= 40000000
learning_rate=4.5e-5
k = np.prod(x[:, :5], axis=1) 

plt.close('all')

plt.ion()

plt.figure(figsize=(10, 7))

def plot(X, y_true, X_train, y_pred):
    plt.scatter(k, y_true, c='blue', s=30, label='True Prices')  # Use for the product of features
    plt.scatter(k, y_pred, c='red', s=30, label='Predicted Prices')  # Use for the product of features
    #plt.scatter(x[:,3], y_true, c='blue', s=30, label='True Prices')  # Use the first feature for x-axis or any other
    #plt.scatter(x[:,3], y_pred, c='red', s=30, label='Predicted Prices')  # Use the first feature for predictions
    plt.xlabel('Input')
    plt.ylabel('Price ($)')
    plt.title('Training Data vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.show()
    plt.pause(0.1)

def meanSquaredLoss(Y_pred, Y_train, m_train):
    loss = (1/(2 * m_train))*np.sum((Y_pred - Y_train) ** 2)
    return loss

def train_lr(no_of_iterations, learning_rate, X_train, y_train):
    # Initialize weights and bias
    W = np.random.rand(5) * 0.01
    b = 0
    loss_history = []

    for i in range(no_of_iterations):
        # Predictions 
        y_pred = np.matmul(X_train[:, :5], W) + b

        # Calculate loss
        loss = meanSquaredLoss(y_pred, y_train, m_train)
        loss_history.append(loss)

        # Gradients 
        err = y_pred - y_train
        grad_W = np.matmul(X_train[:, :5].T, err) / m_train  # Gradient for weights
        grad_b = np.sum(err) / m_train  # Gradient for bias

        # Update weights and bias 
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b

        if i % 50 == 0:
            plt.clf()  # Clear the current figure
            print(f"Iterations {i}: Loss = {loss}")
            plot(X_train, y_train, X_train, y_pred)

    return W, b, loss_history

# Update accordingly
W, b, loss_history = train_lr(no_of_iterations, learning_rate, X_train=x, y_train=p)

plt.plot()

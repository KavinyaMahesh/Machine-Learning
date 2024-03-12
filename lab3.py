import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def matrix_operations(data):
    # A1. Matrix Operations
    A = data.iloc[:, 1:4].to_numpy()
    C = data.iloc[:, 4].to_numpy()

    # A1. Activities
    dimensionality = A.shape[1]
    num_vectors = A.shape[0]
    rank_A = np.linalg.matrix_rank(A)
    pseudo_inverse_A = np.linalg.pinv(A)
    cost_vector = np.dot(pseudo_inverse_A, C)

    return dimensionality, num_vectors, rank_A, cost_vector

def calculate_model_vector(A, C, reg_term=1e-5):
    # Calculate pseudo-inverse with regularization
    pseudo_inverse_A = np.linalg.pinv(A.T @ A + reg_term * np.eye(A.shape[1])) @ A.T @ C
    return pseudo_inverse_A

def customer_classification(data):
    # A3. Customer Classification
    data['Customer_Class'] = np.where(data['Payment (Rs)'] > 200, 'RICH', 'POOR')
    return data[['Customer', 'Payment (Rs)', 'Customer_Class']]

def stock_price_analysis(stock_data):
    # A4. Stock Price Analysis
    mean_price = stock_data['Price'].mean()
    variance_price = stock_data['Price'].var()

    wednesday_prices = stock_data[stock_data['Day'] == 'Tue']['Price']
    mean_wednesday_prices = wednesday_prices.mean()

    april_prices = stock_data[stock_data['Month'] == 'Jun']['Price']
    mean_april_prices = april_prices.mean()

    loss_probability = len(stock_data[lambda x: x['Chg%'] < 0]) / len(stock_data)
    wednesday_profit_probability = len(wednesday_prices[lambda x: x > 0]) / len(wednesday_prices)
    conditional_profit_probability = len(wednesday_prices[lambda x: x > 0]) / len(wednesday_prices)

    return mean_price, variance_price, mean_wednesday_prices, mean_april_prices, loss_probability, wednesday_profit_probability, conditional_profit_probability

def plot_stock_data(stock_data):
    # Scatter plot
    plt.scatter(stock_data['Day'], stock_data['Chg%'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Chg%')
    plt.show()

def main():
    # Load data
    data_purchase = pd.read_excel(r"C:\Users\sripriya konjarla\Downloads\Kavinya\Lab Session1 Data.xlsx", sheet_name="Purchase data")
    stock_data = pd.read_excel(r"C:\Users\sripriya konjarla\Downloads\Kavinya\Lab Session1 Data.xlsx", sheet_name="IRCTC Stock Price")

    # A1 Results
    dimensionality, num_vectors, rank_A, cost_vector = matrix_operations(data_purchase)
    print(f"Dimensionality of the vector space: {dimensionality}")
    print(f"How many vectors exist in this vector space: {num_vectors}")
    print(f"Rank of Matrix A: {rank_A}")
    print("\nCost of each product available for sale:")
    print(cost_vector)

    # A2
    pseudo_inverse_A = calculate_model_vector(data_purchase.iloc[:, 1:4].to_numpy(), data_purchase.iloc[:, 4].to_numpy())
    print("\nPseudo-Inverse of A:")
    print(pseudo_inverse_A)

    # A3 Results
    customer_data_classification = customer_classification(data_purchase)
    print("\nCustomer Data with Classification:")
    print(customer_data_classification)

    # A4 Results
    mean_price, variance_price, mean_wednesday_prices, mean_april_prices, loss_probability, wednesday_profit_probability, conditional_profit_probability = stock_price_analysis(stock_data)
    print("\nStock Price Analysis:")
    print(f"Mean Price: {mean_price}")
    print(f"Variance of Price: {variance_price}")
    print(f"Mean Price on Tuesday: {mean_wednesday_prices}")
    print(f"Mean Price in June: {mean_april_prices}")
    print(f"Probability of Making a Loss: {loss_probability}")
    print(f"Probability of Making a Profit on Tuesday: {wednesday_profit_probability}")
    print(f"Conditional Probability of Making Profit on Tuesday: {conditional_profit_probability}")

    # A4 Plot
    plot_stock_data(stock_data)

if __name__ == "__main__":
    main()

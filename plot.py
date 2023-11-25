import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_results(csv_file_path):
    # Read data from CSV file
    data = np.genfromtxt(csv_file_path, delimiter=',', names=True)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data['Iteration'], data['Reward'], label='Reward')
    plt.plot(data['Iteration'], data['Critic Loss'], label='Critic Loss')
    plt.plot(data['Iteration'], data['Actor Loss'], label='Actor Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    csv_file_path = 'log_data.csv'
    plot_results(csv_file_path)

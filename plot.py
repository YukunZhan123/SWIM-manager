import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_results(csv_file_path, window_size=200):
    # Read data from CSV file
    data = np.genfromtxt(csv_file_path, delimiter=',', names=True)

    cn = data.dtype.names[3:4]

    num_rows = len(data)
    num_windows = num_rows // window_size
    means = {c: [] for c in cn}
    its = []
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window_data = data[start_idx:end_idx]
        
        for c in cn:
            c_mean = np.mean(window_data[c])
            means[c].append(c_mean)
        window_it = start_idx + (window_size // 2)
        its.append(window_it)
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.title('Critic Loss')
    #plt.yscale('log')
    for c in cn:
        plt.plot(its, means[c], label=c)
    #plt.plot(data['Iteration'], data['Reward'], label='Reward')
    #plt.plot(data['Iteration'], data[cn[2]], label='Critic Loss')
    #plt.plot(data['Iteration'], data[cn[3]], label='Actor Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    csv_file_path = 'log_data.csv'
    plot_results(csv_file_path)

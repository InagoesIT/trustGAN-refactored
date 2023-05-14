import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_execution_time(path_to_load):
    time_data = np.load("{}/execution_data.npy".format(path_to_load), allow_pickle=True)
    time_data = time_data.item()
    print(time_data)

    for epoch, time in enumerate(time_data["time"]):
        plt.plot(epoch, time)
    plt.xlabel("Epoch")
    plt.ylabel("Execution time in minutes")
    plt.grid()
    plt.savefig("{}/time-plot.png".format(path_to_load))
    plt.clf()
    
    
plot_execution_time(sys.argv[1])
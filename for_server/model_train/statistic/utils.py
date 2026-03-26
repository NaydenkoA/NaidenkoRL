import matplotlib.pyplot as plt

def plot(series, x, y, name = None, times = None, to_date = False, labels = None):
    n = len(series)

    if times is not None:
        plt.figure(figsize=(12, 5))
        if labels is not None:
            for i in range(n):
                plt.plot(times, series[i], label = labels[i])
            plt.legend()
        else:
            for i in range(n):
                plt.plot(times, series[i])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True)
        plt.show()
    else:
        plt.figure(figsize=(12, 5))
        if labels is not None:
            for i in range(n):
                plt.plot(series[i], label = labels[i])
            plt.legend()
        else:
            for i in range(n):
                plt.plot(series[i])
        plt.legend()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True)
        plt.show()
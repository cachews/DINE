import mlflow
import matplotlib.pyplot as plt

def plot_test_pred(args, true, pred, series_id=0, show=False):
    
    plt.figure(figsize=(15, 6))
    plt.plot(true, label="True")
    plt.plot(pred, label="Predicted")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title(args.name + f" - Test Prediction for Series ID: {series_id}")
    plt.legend()
    
    mlflow.log_figure(plt.gcf(), f"test_prediction_plot_series_{series_id}.png")
    
    if show:
        plt.show()

    plt.close()
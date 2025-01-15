from zenml.client import Client

from pipelines.training_pipeline import train_pipeline


if __name__ == "__main__":
    # Path to the data
    data_path = "data/olist_customers_dataset.csv"

    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    
    # Run the pipeline
    train_pipeline(data_path)

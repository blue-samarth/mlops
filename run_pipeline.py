from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Path to the data
    data_path = "data\olist_customers_dataset.csv"
    
    # Run the pipeline
    train_pipeline(data_path)
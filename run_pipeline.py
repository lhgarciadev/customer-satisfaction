from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    #train_pipeline(data_path="file:C:\Users\lhgar\AppData\Roaming\zenml\local_stores\333676a5-9efd-4d52-8d48-7698d067257e\mlruns")
    # Ingest the data
    data = ingest_data()

    # Clean the data
    x_train, x_test, y_train, y_test = clean_data(data=data)

    # Define model configuration
    config = ModelNameConfig(model_name="lightgbm", fine_tuning=False)

    # Train the model
    model = train_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, config=config)

    # Evaluate the model
    mse, rmse = evaluation(model=model, x_test=x_test, y_test=y_test)

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )

from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from steps.config import ModelNameConfig

if __name__ == "__main__":
    # Ingest data
    raw_data = ingest_data()

    # Clean data
    x_train, x_test, y_train, y_test = clean_data(raw_data)

    # Specify the model configuration
    model_config = ModelNameConfig(model_name="randomforest", fine_tuning=True)

    # Train model
    trained_model = train_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, config=model_config)

    # Evaluate model
    r2_score, rmse = evaluation(model=trained_model, x_test=x_test, y_test=y_test)

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )

from .nodes_ml.preprocess_images import create_transform_img_test, create_transform_img_train
from kedro.pipeline import Pipeline, node
from .nodes_ml.data_setup import create_dataloaders
from .nodes_ml.model import build_model
from .nodes_ml.train_model import train


def create_ml_pipeline(**kwargs):
    pipeline_create_dataloaders = Pipeline(
        [
            node(
                func=create_transform_img_train,
                inputs=["size"],
                outputs=["train_data_transform"]
            ),

            node(
                func=create_transform_img_test,
                inputs=["size"],
                outputs=["test_data_transform"]
            ),

            node(
                func=create_dataloaders,
                inputs=dict(train_dir="train_dir", test_dir="test_dir",
                            train_transform="train_data_transform", test_transform="test_data_transform"),
                outputs=["train_dataloader", "test_dataloader", "id_to_class"]
            )

        ]
    )

    pipeline_training = Pipeline(
        [
            node(
                func=build_model,
                inputs=dict(input_dim="params:model_parameters:input_dim", hidden_dim="hidden_dim",
                            output_dim="output_dim", device="device"),
                outputs="pt_model"
            ),

            node(
                func=train,
                inputs=dict(model="pt_model", train_dataloader="train_dataloader",
                            test_dataloader="test_dataloader", epochs="epochs"),
                outputs=["pt_model"]

            )
        ]
    )

    return pipeline_create_dataloaders + pipeline_training

from .nodes_ml.preprocess_images import create_img_test_transform, create_img_train_transform
from kedro.pipeline import Pipeline, node
from .nodes_ml.data_setup import create_dataloaders
from .nodes_ml.model import build_model
from .nodes_ml.train_model import train

import re


def create_ml_pipeline(**kwargs):
    pipeline_create_dataloaders = Pipeline(
        [
            node(
                func=create_img_train_transform,
                inputs=dict(size="params:size"),
                # inputs=["size"],
                outputs="train_data_transform"
            ),

            node(
                func=create_img_test_transform,
                inputs=["params:size"],
                outputs="test_data_transform"
            ),

            node(
                func=create_dataloaders,
                inputs={
                    "train_dir": "params:train_dir",
                    "test_dir": "params:test_dir",
                    "train_transform": "train_data_transform",
                    "test_tranform": "test_data_transform",
                    "batch_size": "params:batch_size",
                    "num_workers": "params:num_workers"
                },
                outputs=["train_dataloader", "test_dataloader", "id_to_class"]
            )

        ]
    )

    pipeline_training = Pipeline(
        [
            node(
                func=build_model,
                # inputs=dict(input_dim="params:model_parameters:input_dim", hidden_dim="hidden_dim",
                #             output_dim="output_dim"),
                inputs={"input_dim": "params:model_parameters.input_dim",
                        "hidden_dim": "params:model_parameters.hidden_dim",
                        "output_dim": "params:model_parameters.output_dim"},
                outputs="pt_model"
            ),

            node(
                func=train,
                # inputs=dict(model="pt_model", train_dataloader="train_dataloader",
                #             test_dataloader="test_dataloader", epochs="epochs"),

                inputs={
                    "model": "pt_model",
                    "train_dataloader": "train_dataloader",
                    "test_dataloader": "test_dataloader",
                    "epochs": "params:training_loop_parameters.epochs",
                    "optimizer_kwargs": "params:training_loop_parameters.optimizer_params"
                },
                outputs="pt_model_trained"

            )
        ]
    )

    return pipeline_create_dataloaders + pipeline_training

from kedro.pipeline import node, Pipeline
from .nodes.create_data import get_data


def create_etl_pipeline(**kwargs):
    pipeline_etl = Pipeline(
        [
            node(
                func=get_data,
                inputs=["params:data_url", "params:storage_path"],
                outputs=None
            )
        ]
    )
    return pipeline_etl

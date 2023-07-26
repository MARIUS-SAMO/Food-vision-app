"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from food_vision_app.pipelines.etl_app.pipeline import create_etl_pipeline
from food_vision_app.pipelines.ml_app.pipeline import create_ml_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    etl_pipeline = create_etl_pipeline()
    ml_pipeline = create_ml_pipeline()

    return {
        "etl_pipeline": etl_pipeline,
        "ml_pipeline": ml_pipeline,
        "__default__": etl_pipeline + ml_pipeline
    }

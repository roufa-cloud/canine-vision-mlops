"""
This script registers the best performing model in MLflow for deployment.
It loads best model from evaluations, wraps it in MLflow format,
logs it with metadata (accuracy, hyperparameters) and registers it in MLflow model registry
Run this ONCE after prepare_deployment_artifacts.py completes."""

import json
import sys
import numpy as np
from pathlib import Path
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature
import tensorflow as tf
import yaml
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_best_model_info():
    """ load best model metadata from evaluation"""
    info_path = project_root/"artefacts"/"evaluations"/"best_model_info.json"
    
    if not info_path.exists():
        raise FileNotFoundError(
            f"Best model info not found at {info_path}. "
            "Please run: python scripts/prepare_deployment_artifacts.py"
        )
    
    with open(info_path) as f:
        return json.load(f)


def load_training_config():
    """ load training configuration from config file"""
    
    config_path = project_root/"config"/"default.yaml"
    if not config_path.exists():
        print(f"[ALERT] Config file not found at {config_path}")
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def register_model():
    """ main function to register model in MLflow """

    print("=" * 60)
    print("REGISTERING MODEL IN MLFLOW")
    print("=" * 60)

    print("\n1. Loading best model info...")
    try:
        model_info = load_best_model_info()
    except FileNotFoundError as e:
        print(f"[ALERT] {e}")
        return
    
    model_name = model_info["model_name"]
    checkpoint_file = model_info["checkpoint_file"]
    val_accuracy = model_info["val_accuracy"]
    
    print(f"   Model: {model_name}")
    print(f"   Checkpoint: {checkpoint_file}")
    print(f"   Val Accuracy: {val_accuracy:.4f}")
    
    print("\n2. Loading model checkpoint...")
    checkpoint_path = project_root/"artefacts"/"checkpoints"/checkpoint_file
    
    if not checkpoint_path.exists():
        print(f"[ALERT] Checkpoint not found: {checkpoint_path}")
        return
    
    model = tf.keras.models.load_model(checkpoint_path)
    print(f"   Loaded {checkpoint_path.name}")
    
    print("\n3. Loading training configuration...")
    config = load_training_config()

    print("\n4. Setting up MLflow...")
    mlflow_dir = project_root/"mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(str(mlflow_dir.resolve().as_uri()))

    experiment_name = "dog-breed-classification"
    mlflow.set_experiment(experiment_name)
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   Experiment: {experiment_name}")
    
    print("\n5. Logging model to MLflow...")
    with mlflow.start_run(run_name=f"{model_name}_deployment") as run:
        params = {
            "model_architecture": model_name,
            "img_size": config.get("dataset", {}).get("img_size", 224),
            "batch_size": config.get("dataset", {}).get("batch_size", 32),
            "num_classes": 120,
        }
        
        if "resnet" in model_name.lower():
            model_config = config.get("models", {}).get("resnet50", {})
        elif "efficientnet" in model_name.lower():
            model_config = config.get("models", {}).get("efficientnetb0", {})
        elif "fusion" in model_name.lower():
            model_config = config.get("models", {}).get("fusion", {})
        else:
            model_config = {}
        
        params.update({f"model_{k}": v for k, v in model_config.items()})
        
        mlflow.log_params(params)

        metrics = {
            "val_accuracy": val_accuracy,
            "val_loss": model_info.get("val_loss", 0.0),
            "val_top_5_accuracy": model_info.get("val_top_5_accuracy", 0.0),
        }
        mlflow.log_metrics(metrics)

        is_fusion = "fusion" in model_name.lower()
        if is_fusion:
            input_example = {
                "resnet50": np.random.rand(1, 224, 224, 3).astype(np.float32),
                "efficientnetb0": np.random.rand(1, 224, 224, 3).astype(np.float32),
            }
        else:
            input_example = np.random.rand(1, 224, 224, 3).astype(np.float32)

        output_example = model.predict(input_example, verbose=0)
        signature = infer_signature(input_example, output_example)
     
        mlflow.keras.log_model(
            model=model,
            name="model",
            registered_model_name="dog-breed-classifier",
            signature=signature
        )
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

        if (project_root/"config"/"default.yaml").exists():
            mlflow.log_artifact(
                str(project_root/"config"/"default.yaml"),
                artifact_path="config"
            )
        
        print(f"   Model logged successfully")
        print(f"   Run ID: {run.info.run_id}")
    
    print("\n6. Registered Model Information:")
    client = mlflow.tracking.MlflowClient()
    
    try:
        registered_model = client.get_registered_model("dog-breed-classifier")
        latest_versions = registered_model.latest_versions
        
        if latest_versions:
            latest_version = latest_versions[0]
            print(f"   Model Name: dog-breed-classifier")
            print(f"   Version: {latest_version.version}")
            print(f"   Stage: {latest_version.current_stage}")
            print(f"   Run ID: {latest_version.run_id}")
    except Exception as e:
        print(f"  [ALERT] Could not retrieve registered model info: {e}")
    
    print("\n7. Saving MLflow model URI...")
    mlflow_info = {
        "model_name": "dog-breed-classifier",
        "run_id": run.info.run_id,
        "model_uri": f"runs:/{run.info.run_id}/model",
        "registered_at": run.info.end_time,
    }
    
    mlflow_info_path = project_root/"artefacts"/"mlflow_model_info.json"
    with open(mlflow_info_path, "w") as f:
        json.dump(mlflow_info, f, indent=2)
    
    print(f"   Saved MLflow info to {mlflow_info_path.name}")

    print("\n" + "=" * 60)
    print("MODEL SUCCESSFULLY REGISTERED IN MLFLOW")
    print("=" * 60)
    
    print("\nTo load the model:")
    print(f"  import mlflow")
    print(f'  model = mlflow.keras.load_model("runs:/{run.info.run_id}/model")')
    
    print("\nTo view MLflow UI:")
    print(f"  cd {project_root}")
    print(f"  mlflow ui")
    print(f"  # Then open http://localhost:5000")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        register_model()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

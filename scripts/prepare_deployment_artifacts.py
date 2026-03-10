""" 
This script prepare deployment artifacts for best model
It identifies the best model from evaluation results, copies necessary files to a deployment directory,
creates class_names.json from datasets and validates all required artifacts exist.
"""

import json
from pathlib import Path
import tensorflow_datasets as tfds


def get_project_root():
    """ get project root directory  
    Returns:
        path to project root directory
    """
    return Path(__file__).parent.parent


def create_class_names_file(output_path):
    """ 
    create class_names.json from stanford dogs dataset  
    Args:
        output_path(path): where to save class_names.json
    """
    
    print("   Loading Stanford Dogs dataset to extract class names...")
    _, ds_info = tfds.load(
        "stanford_dogs",
        split="train",
        with_info=True,
        download=False, 
    )
    
    # clean class names
    raw_class_names = ds_info.features["label"].names
    class_names = [name.split("-", 1)[1] for name in raw_class_names]
    
    # save JSON
    with open(output_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"   Created class_names.json with {len(class_names)} classes")


def prepare_deployment_artifacts():
    """ function to prepare all deployment artifacts """

    project_root = get_project_root()
    artefacts_dir = project_root/"artefacts"
    
    print("=" * 60)
    print("PREPARING DEPLOYMENT ARTIFACTS")
    print("=" * 60)
    
    # 1- check if best_model_info.json exists
    best_model_info_path = artefacts_dir/"evaluations"/"best_model_info.json"
    
    if not best_model_info_path.exists():
        print("\nbest_model_info.json not found!")
        print("   Please run notebook 05_evaluation_and_comparison.ipynb first.")
        return
    
    # load best model info
    with open(best_model_info_path) as f:
        best_model_info = json.load(f)
    
    print(f"\n1. Best Model Information")
    print(f"   Model: {best_model_info['model_name']}")
    print(f"   Checkpoint: {best_model_info['checkpoint_file']}")
    print(f"   Val Accuracy: {best_model_info['val_accuracy']:.4f}")
    
    # 2- verify checkpoint
    checkpoint_path = artefacts_dir/"checkpoints"/best_model_info["checkpoint_file"]
    if not checkpoint_path.exists():
        print(f"\n[ALERT] Checkpoint not found: {checkpoint_path}")
        return
    print(f"\n2. Checkpoint exists: {checkpoint_path.name}")
    
    # 3- create class_names.json
    class_names_path = artefacts_dir/"class_names.json"
    if not class_names_path.exists():
        print("\n3. Creating class_names.json...")
        try:
            create_class_names_file(class_names_path)
        except Exception as e:
            print(f"   [ALERT] Failed to create class_names.json: {e}")
            print("   Please ensure tensorflow_datasets is installed and dataset is downloaded.")
            return
    else:
        print("\n3. class_names.json already exists")
    
    # 4- verify required artifacts
    print("\n4. Verifying deployment artifacts:")
    required_files = {
        "Best model checkpoint": checkpoint_path,
        "Class names": class_names_path,
        "Best model metadata": best_model_info_path,
    }
    
    all_exist = True
    for name, path in required_files.items():
        exists = path.exists()
        status = "[OKAY]" if exists else "[ALERT]"
        print(f"   {status} {name}: {path.name}")
        if not exists:
            all_exist = False
    
    # 5- Summary
    print("\n" + "=" * 60)
    if all_exist:
        print("ALL DEPLOYMENT ARTIFACTS READY")
    else:
        print("[ALERT] SOME ARTIFACTS MISSING")
        print("\nPlease:")
        print("  1. Train models (notebooks 02-04)")
        print("  2. Run evaluation (notebook 05)")
        print("  3. Re-run this script")
    print("=" * 60)


if __name__ == "__main__":
    prepare_deployment_artifacts()

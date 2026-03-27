import json
import os
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess


def get_project_root():
    """ get project root directory  
    Returns:
        path to the project root directory
    """
    return Path(__file__).parent.parent.parent


def load_class_names(class_names_path=None):
    """ 
    Load class names from JSON file  
    Args:
        class_names_path(str): path to class_names.json, if none it will use default path
    Returns:
        list of class names
    """
    if class_names_path is None:
        project_root = get_project_root()
        class_names_path = project_root/"artefacts"/"class_names.json"
    
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            f"Class names file not found at {class_names_path}. "
            "Please run: python scripts/prepare_deployment_artifacts.py"
        )
    
    with open(class_names_path) as f:
        class_names = json.load(f)
    
    return class_names


def load_best_model_info(info_path=None):
    """ 
    load best model metadata from evaluation phase  
    Args:
        info_path(str): path to best_model_info.json, if none it will use default path
    Returns:
        dict: best model metadata
    """
    if info_path is None:
        project_root = get_project_root()
        info_path = project_root/"artefacts"/"evaluations"/"best_model_info.json"
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"Best model info not found at {info_path}. "
            "Please run notebook 05_evaluation_and_comparison.ipynb, then "
            "python scripts/prepare_deployment_artifacts.py"
        )
    
    with open(info_path) as f:
        info = json.load(f)
    
    return info


def get_preprocess_func(model_type):
    """ 
    Get preprocessing function based on model type 
    Args:
        model_type(str): type of the model (e.g. "resnet50", "efficientnetb0")
    Returns:
        preprocessing function 
    """
    model_type_lower = model_type.lower()
    if "resnet" in model_type_lower:
        return resnet_preprocess
    elif "efficientnet" in model_type_lower:
        return effnet_preprocess
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Expected 'resnet50' or 'efficientnetb0'. "
            f"For fusion models, call this function twice (once per backbone)."
        )


def format_breed_name(breed_name):
    """ 
    format breed name for display (e.g. "golden_retriever" -> "Golden Retriever") 
    Args:
        breed_name(str): breed name to format
    Returns:
        formatted breed name
    """
    return breed_name.replace("_", " ").title()


def validate_image_file(file_path):
    """
    Validate if file is an image 
    Args:
        file_path(str): path to the file to validate
    Returns:
        bool: True if file is a valid image, False otherwise
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    if not os.path.exists(file_path):
        return False
    
    ext = Path(file_path).suffix.lower()
    return ext in valid_extensions

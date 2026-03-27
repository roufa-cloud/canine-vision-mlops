import os
import json
from pathlib import Path
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from PIL import Image

from .utils import (
    get_project_root,
    load_class_names,
    load_best_model_info,
    get_preprocess_func,
    format_breed_name,
    validate_image_file,
)

class DogBreedPredictor:
    """ 
    Predictor for dog breed classification 
    It loads a trained model, preprocesses input images and returns predictions
    It automatically loads from MLflow registry first then falls back to local checkpoint and it 
    can handle both single and batch predictions.
    It supports ResNet50, EfficientNetB0 and Fusion models with appropriate preprocessing
     Example usage:
        predictor = DogBreedPredictor()
        predictor = DogBreedPredictor(prefer_mlflow=False) # force loading from checkpoint
        prediction = predictor.predict("path/to/dog.jpg", top_k=5)
        print(f"Top prediction: {prediction['top_prediction']}")
        print(f"Confidence: {prediction['top_confidence']:.2%}")
    """
    
    def __init__(self, model_path=None, class_names_path=None, img_size=224, 
                 use_best_model=True, prefer_mlflow=True):
        """ Initialize the predictor by loading the model and setting up preprocessing
        Args:
            model_path(str): path to model checkpoint, if None and use_best_model=True, 
                            loads best model from evaluation phase
            class_names_path(str): path to class names file
            img_size(int): size of the input images
            use_best_model(bool): if True, use best model from evaluation
            prefer_mlflow(bool): if True, tries MLflow first then falls back to checkpoint loading
        """
        self.img_size = img_size
        self.project_root = get_project_root()
        self.class_names = load_class_names(class_names_path)
        self.num_classes = len(self.class_names)
        self.model_metadata = {}
        self.model_source = None

        # load model
        if prefer_mlflow and model_path is None:
            try:
                self._load_from_mlflow()
            except (FileNotFoundError, ImportError, Exception) as e:
                print(f"==> MLflow loading unavailable: {type(e).__name__}")
                print(f"    Falling back to checkpoint loading...")
                self._load_from_checkpoint(model_path, use_best_model)
        else:
            self._load_from_checkpoint(model_path, use_best_model)
        
        # preprocessing function
        self.is_fusion = "fusion" in self.model_name.lower()
        if self.is_fusion:
            self.preprocess_resnet = get_preprocess_func("resnet50")
            self.preprocess_effnet = get_preprocess_func("efficientnetb0")
        else:
            self.preprocess_fn = get_preprocess_func(self.model_name)
        
        # summary
        print(f"==> Model loaded successfully")
        print(f"    Source: {self.model_source}")
        print(f"    Model: {self.model_name}")
        print(f"    Classes: {self.num_classes}")
        print(f"    Image size: {self.img_size}x{self.img_size}")
        
        val_acc = self.model_metadata.get('val_accuracy')
        if val_acc is not None:
            print(f"    Val accuracy: {val_acc:.4f}")
    
    
    def _load_from_mlflow(self):
        """ load model from MLflow registry """
        
        mlflow_info_path = self.project_root/"artefacts"/"mlflow_model_info.json"
        if not mlflow_info_path.exists():
            raise FileNotFoundError(
                f"MLflow model info not found at {mlflow_info_path} "
                "Run: python scripts/register_model_mlflow.py")
        
        with open(mlflow_info_path) as f:
            mlflow_info = json.load(f)

        mlflow_dir = self.project_root/"mlruns"
        if not mlflow_dir.exists():
            raise FileNotFoundError(
                f"MLflow tracking directory not found at {mlflow_dir} "
                "Run: python scripts/register_model_mlflow.py"
            )
        
        mlflow.set_tracking_uri(mlflow_dir.as_uri()) 
        model_uri = mlflow_info["model_uri"]
        print(f"Loading from MLflow: {model_uri}")
        
        self.model = mlflow.keras.load_model(model_uri)
        run_id = mlflow_info["run_id"]
        client = mlflow.tracking.MlflowClient()
        
        try:
            run = client.get_run(run_id)
            model_architecture = run.data.params.get("model_architecture")
            
            if model_architecture:
                self.model_name = model_architecture
            else:
                self.model_name = mlflow_info["model_name"]
            
            self.model_source = "mlflow"
            self.model_metadata = {
                "model_name": self.model_name,
                "run_id": run_id,
                "val_accuracy": run.data.metrics.get("val_accuracy"),
                "val_loss": run.data.metrics.get("val_loss"),
                "val_top_5_accuracy": run.data.metrics.get("val_top_5_accuracy"),
            }
            
        except Exception as e:
            print(f"==> Could not fetch MLflow run metadata: {e}")
            print("    Using fallback model name detection...")
            self.model_name = mlflow_info["model_name"]
            self.model_source = "mlflow"
            self.model_metadata = {"model_name": self.model_name}

    
    def _load_from_checkpoint(self, model_path, use_best_model):
        """ 
        Load model from checkpoint file  
        Args:
            model_path(str): path to model checkpoint, if None, uses best model
            use_best_model(bool): if True, use best model from evaluation
        
        """
        if model_path is None and use_best_model:
            best_model_info = load_best_model_info()
            self.model_name = best_model_info["model_name"]
            checkpoint_file = best_model_info["checkpoint_file"]
            model_path = self.project_root/"artefacts"/"checkpoints"/checkpoint_file
            self.model_metadata = best_model_info
        else:
            # Try to load best_model_info as fallback for model_name
            best_model_info = None
            try:
                best_model_info = load_best_model_info()
            except:
                pass
            
            if best_model_info and "model_name" in best_model_info:
                self.model_name = best_model_info["model_name"]
                self.model_metadata = best_model_info
            else:
                self.model_name = Path(model_path).stem if model_path else "unknown"
                self.model_metadata = {}

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please run: python scripts/prepare_deployment_artifacts.py")
        print(f"Loading from checkpoint: {Path(model_path).name}")
        self.model = tf.keras.models.load_model(model_path)
        self.model_source = "checkpoint"
    

    def preprocess_image(self, image):
        """ preprocess an image for model input   
        Args:
            images: str (file path) or PIL.Image or numpy array
        Returns:
            preprocessed image ready for model input
        """
        if isinstance(image, str):
            if not validate_image_file(image):
                raise ValueError(f"Invalid image file: {image}")
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Image must be RGB (H, W, 3). Got shape: {image.shape}"
            )

        image = tf.image.resize(image, (self.img_size, self.img_size))
        image = image.numpy()
        
        if self.is_fusion:
            image_resnet = self.preprocess_resnet(np.copy(image))
            image_effnet = self.preprocess_effnet(np.copy(image))
            return {
                "resnet50": np.expand_dims(image_resnet, axis=0),
                "efficientnetb0": np.expand_dims(image_effnet, axis=0),
            }
        else:
            image = self.preprocess_fn(image)
            return np.expand_dims(image, axis=0)
    

    def predict(self, image, top_k=5):
        """ 
        Predict dog breed from an image  
        Args:
            image: str (file path) or PIL.Image or numpy array
            top_k: no. of top predictions to return
        Returns:
            dict with keys:
                - top_prediction: breed name(str)
                - top_confidence: confidence of top prediction (float)
                - top_k_predictions: list of (breed name, confidence) for top k predictions
                - all_probabilities: array of probabilities for all classes  (np.ndarray)
        """
        preprocessed = self.preprocess_image(image)
        predics = self.model.predict(preprocessed, verbose=0)
        probabs = predics[0] # remove batch dimension
        top_k_indices = np.argsort(probabs)[-top_k:][::-1]
        top_k_predics = [
            (format_breed_name(self.class_names[idx]), float(probabs[idx]))
            for idx in top_k_indices]
        return {
            "top_prediction": top_k_predics[0][0],
            "top_confidence": top_k_predics[0][1],
            "top_k_predictions": top_k_predics,
            "all_probabilities": probabs}
    

    def predict_batch(self, images, top_k=5):
        """ predict on multiple images (batch processing) 
        Args:
            images: list of str (file paths) or PIL.Images or numpy arrays
            top_k: no. of top predictions to return for each image
        Returns:
            list of dicts, each dict has same format as predict() output
        """
        results = []
        for image in images:
            result = self.predict(image, top_k=top_k)
            results.append(result)
        return results
    

    def get_model_info(self):
        """" return model information and metadata for display 
        Returns:
            dict with model name, source, num_classes, img_size and any additional metadata"""
        return {
            "model_name": self.model_name,
            "model_source": self.model_source,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            **self.model_metadata,
        }


def create_predictor(model_path=None, use_best_model=True):
    """ 
    Function to create a DogBreedPredictor instance
    Args:
        model_path(str): path to model checkpoint
        use_best_model(bool): if True, use best model from evaluation
    Returns:
        DogBreedPredictor instance
    """
    return DogBreedPredictor(
        model_path=model_path,
        use_best_model=use_best_model,
    )
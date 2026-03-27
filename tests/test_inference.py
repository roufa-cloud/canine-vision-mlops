"""
This script tests preprocessing logic, prediction accuracy, error handling
and predictor with both MLflow and checkpoint loading

"""
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import utils, predictor


class TestUtils(unittest.TestCase):
    """ Test utility functions """
    
    def test_get_project_root(self):
        """ Test project root detection """
        root = utils.get_project_root()
        self.assertIsInstance(root, Path)
        self.assertTrue(root.exists())
        self.assertTrue((root/"src").exists())
    
    def test_format_breed_name(self):
        """ Test breed name formatting """
        test_cases = [
            ("labrador_retriever", "Labrador Retriever"),
            ("golden_retriever", "Golden Retriever"),
            ("german_shepherd", "German Shepherd"),
            ("poodle", "Poodle"),
        ]
        for input_name, expected in test_cases:
            self.assertEqual(utils.format_breed_name(input_name), expected)
    
    def test_validate_image_file(self):
        """ Test image file validation """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        
        try:
            self.assertTrue(utils.validate_image_file(temp_path))
            self.assertFalse(utils.validate_image_file("/nonexistent.jpg"))
        finally:
            Path(temp_path).unlink()
    
    def test_get_preprocessing_function(self):
        """ Test preprocessing function selection """
        resnet_fn = utils.get_preprocess_func("resnet50")
        self.assertTrue(callable(resnet_fn))
        
        effnet_fn = utils.get_preprocess_func("efficientnetb0")
        self.assertTrue(callable(effnet_fn))
        
        with self.assertRaises(ValueError):
            utils.get_preprocess_func("unknown_model")


class TestPredictor(unittest.TestCase):
    """ Test DogBreedPredictor class """
    
    def setUp(self):
        """Set up test fixtures """
        self.mock_class_names = ["labrador_retriever", "golden_retriever", "poodle"]
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([[0.7, 0.2, 0.1]])
    
    @patch('src.inference.predictor.load_class_names')
    @patch('src.inference.predictor.load_best_model_info')
    @patch('src.inference.predictor.tf.keras.models.load_model')
    def test_init_checkpoint(self, mock_load_model, mock_load_info, mock_load_classes):
        """ Test initialization with checkpoint loading """
        mock_load_classes.return_value = self.mock_class_names
        mock_load_model.return_value = self.mock_model
        
        # mock best_model_info to provide valid model name
        mock_load_info.return_value = {
            "model_name": "ResNet50 Fine-Tuned",
            "checkpoint_file": "resnet50.keras",
            "val_accuracy": 0.85
        }
        
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            temp_path = f.name
        
        try:
            pred = predictor.DogBreedPredictor(
                model_path=temp_path,
                use_best_model=False,
                prefer_mlflow=False
            )
            self.assertEqual(pred.model_source, "checkpoint")
        finally:
            Path(temp_path).unlink()
    
    @patch('src.inference.predictor.load_class_names')
    @patch('src.inference.predictor.load_best_model_info')
    @patch('src.inference.predictor.tf.keras.models.load_model')
    def test_preprocess_image(self, mock_load_model, mock_load_info, mock_load_classes):
        """ Test image preprocessing """
        mock_load_classes.return_value = self.mock_class_names
        mock_load_model.return_value = self.mock_model
        
        # mock best_model_info with valid model name
        mock_load_info.return_value = {
            "model_name": "EfficientNetB0 Fine-Tuned",
            "checkpoint_file": "efficientnetb0.keras",
            "val_accuracy": 0.82
        }
        
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            temp_model = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_img = f.name
            Image.new('RGB', (100, 100), 'red').save(temp_img)
        
        try:
            pred = predictor.DogBreedPredictor(
                model_path=temp_model,
                use_best_model=False,
                prefer_mlflow=False
            )
            
            preprocessed = pred.preprocess_image(temp_img)
            self.assertEqual(preprocessed.shape, (1, 224, 224, 3))
        finally:
            Path(temp_model).unlink()
            Path(temp_img).unlink()
    
    @patch('src.inference.predictor.load_class_names')
    @patch('src.inference.predictor.load_best_model_info')
    @patch('src.inference.predictor.tf.keras.models.load_model')
    def test_predict(self, mock_load_model, mock_load_info, mock_load_classes):
        """ Test prediction """
        mock_load_classes.return_value = self.mock_class_names
        mock_load_model.return_value = self.mock_model
        
        # mock best_model_info with valid model name
        mock_load_info.return_value = {
            "model_name": "ResNet50 Fine-Tuned",
            "checkpoint_file": "resnet50.keras",
            "val_accuracy": 0.85
        }
        
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            temp_model = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_img = f.name
            Image.new('RGB', (100, 100), 'blue').save(temp_img)
        
        try:
            pred = predictor.DogBreedPredictor(
                model_path=temp_model,
                use_best_model=False,
                prefer_mlflow=False
            )
            
            result = pred.predict(temp_img, top_k=3)
            
            self.assertIn('top_prediction', result)
            self.assertEqual(result['top_prediction'], "Labrador Retriever")
            self.assertAlmostEqual(result['top_confidence'], 0.7)
            self.assertEqual(len(result['top_k_predictions']), 3)
        finally:
            Path(temp_model).unlink()
            Path(temp_img).unlink()


def run_tests():
    """ Run all tests """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictor))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
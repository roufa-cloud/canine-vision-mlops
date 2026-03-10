"""
This script tests inference pipeline locally.
It first loads the best model, then it runs prediction on a sample image 
(or user-provided image) and displays the results in a readable format."""

import argparse
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.inference.predictor import create_predictor


def get_sample_image():
    """ return path to a local sample image for testing inference"""

    output_path = project_root/"assets"/"sample_images"/"german-shepherd.jpg"
    if not output_path.exists():
        raise FileNotFoundError(f"Sample image not found: {output_path}")
    print(f"Using sample image: {output_path}")
    return str(output_path)


def test_inference(image_path=None, top_k=5):
    """ 
    Test inference pipeline 
    Args:
        image_path(str): path to test image, if None it will use sample image
        top_k(int): number of top predictions to display
    """

    print("=" * 60)
    print("DOG BREED PREDICTOR - INFERENCE TEST")
    print("=" * 60)

    if image_path is None:
        try:
            image_path = get_sample_image()
        except Exception as e:
            print(f"[ALERT] Failed to get sample image: {e}")
            print("  Please provide an image path: python scripts/test_inference.py --image <path>")
            return
    
    if not Path(image_path).exists():
        print(f"[ALERT] Image not found: {image_path}")
        return
    
    print(f"\nTest image: {image_path}")
    print("\nLoading model...")
    try:
        predictor = create_predictor(use_best_model=True)
    except FileNotFoundError as e:
        print(f"\n[ALERT] {e}")
        print("\nPlease ensure:")
        print("  1. Models are trained (notebooks 02-04)")
        print("  2. Evaluation is complete (notebook 05)")
        print("  3. Run: python scripts/prepare_deployment_artifacts.py")
        return
    
    model_info = predictor.get_model_info()
    print(f"\nModel Information:")
    print(f"  Name: {model_info['model_name']}")
    print(f"  Classes: {model_info['num_classes']}")
    val_acc = model_info.get('val_accuracy')
    if val_acc is not None:
        print(f"  Val Accuracy: {val_acc:.4f}")

    print(f"\nRunning prediction...")
    try:
        result = predictor.predict(image_path, top_k=top_k)
    except Exception as e:
        print(f"[ALERT] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-" * 60)
    print("PREDICTION RESULTS")
    print("-" * 60)
    
    print(f"\nTop Prediction: {result['top_prediction']}")
    print(f"   Confidence: {result['top_confidence']:.2%}")
    
    print(f"\nTop {top_k} Predictions:")
    for i, (breed, confidence) in enumerate(result['top_k_predictions'], 1):
        bar_length = int(confidence * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {i}. {breed:30s} {bar} {confidence:.2%}")
    
    print("\n" + "=" * 60)
    print("INFERENCE SUCCESSFULLY COMPLETED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test dog breed inference pipeline")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to dog image (downloads sample if not provided)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)"
    )
    
    args = parser.parse_args()
    test_inference(args.image, args.top_k)


if __name__ == "__main__":
    main()

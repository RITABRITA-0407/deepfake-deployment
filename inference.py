import argparse
import os
import logging
from pathlib import Path
from efficientnet_image_detector import DetectorInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Deepfake Image Detection Inference')
    parser.add_argument('--model', type=str, default='models/final_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto')
    parser.add_argument('--image-size', type=int, default=380,
                        help='Input image size for EfficientNet-B4')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')

    args = parser.parse_args()

    # Validate model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return

    # Initialize detector
    logger.info("Initializing detector...")
    detector = DetectorInference(args.model, device=args.device, image_size=args.image_size)

    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            logger.error(f"Image not found: {args.image}")
            return

        logger.info(f"Predicting on image: {args.image}")
        result = detector.predict(args.image)

        print("\n" + "=" * 50)
        print(f"Image: {args.image}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Real Probability: {result['real_probability']:.4f}")
        print(f"Fake Probability: {result['fake_probability']:.4f}")
        print("=" * 50 + "\n")

    # Folder prediction
    elif args.folder:
        if not os.path.exists(args.folder):
            logger.error(f"Folder not found: {args.folder}")
            return

        logger.info(f"Predicting on folder: {args.folder}")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(args.folder).glob(ext))

        if not image_files:
            logger.warning(f"No images found in {args.folder}")
            return

        results = detector.predict_batch([str(f) for f in image_files])

        print("\n" + "=" * 70)
        print(f"{'Image':<40} {'Verdict':<12} {'Confidence':<15}")
        print("=" * 70)

        real_count = 0
        fake_count = 0

        for result in results:
            if result['success']:
                verdict = result['verdict']
                confidence = result['confidence']
                print(f"{Path(result['image']).name:<40} {verdict:<12} {confidence:.4f}")

                if verdict == 'REAL':
                    real_count += 1
                else:
                    fake_count += 1
            else:
                print(f"{Path(result['image']).name:<40} {'ERROR':<12} -")

        print("=" * 70)
        print(f"Total: {len(results)} | Real: {real_count} | Fake: {fake_count}")
        print("=" * 70 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class DeepfakeImageDetector(nn.Module):
    """
    EfficientNet-B4 based deepfake image detector.
    Binary classifier: Real (0) vs Fake (1)
    """

    def __init__(self, num_classes=2, dropout_rate=0.4, pretrained=True):
        """
        Args:
            num_classes: Number of output classes (default: 2 for Real/Fake)
            dropout_rate: Dropout rate for regularization
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        # Load EfficientNet-B4 backbone
        if pretrained:
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b4(weights=weights)
            logger.info("Loaded EfficientNet-B4 with ImageNet pretraining")
        else:
            self.backbone = efficientnet_b4(weights=None)
            logger.info("Loaded EfficientNet-B4 without pretraining")

        # Get the number of input features for the classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the classifier with custom binary classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)

    def to(self, device):
        """Move model to device"""
        super().to(device)
        self.device = device
        return self

    def get_input_transform(self, image_size=380):
        """Get preprocessing transforms for images"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            self.normalize
        ])

    def predict_single(self, image_path, image_size=380, return_all_probs=False):
        """
        Predict on a single image.

        Args:
            image_path: Path to image file
            image_size: Size to resize image to (default: 380 for EfficientNet-B4)
            return_all_probs: Whether to return all class probabilities

        Returns:
            dict with prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            transform = self.get_input_transform(image_size)
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            # Inference
            self.eval()
            with torch.no_grad():
                logits = self.forward(image_tensor)
                probabilities = torch.softmax(logits, dim=1)

            # Extract predictions
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            result = {
                'verdict': 'DEEPFAKE' if predicted_class == 1 else 'REAL',
                'confidence': confidence,
                'predicted_class': predicted_class,
                'fake_probability': probabilities[0, 1].item(),
                'real_probability': probabilities[0, 0].item(),
                'success': True
            }

            return result

        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            return {
                'verdict': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'success': False
            }

    def predict_batch(self, image_paths, image_size=380):
        """
        Predict on multiple images.

        Args:
            image_paths: List of paths to image files
            image_size: Size to resize images to

        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict_single(image_path, image_size)
            results.append({'image': image_path, **result})
        return results


class DetectorInference:
    """Convenience class for inference with pre-trained model"""

    def __init__(self, model_path, device='auto', image_size=380):
        """
        Initialize detector with trained weights.

        Args:
            model_path: Path to saved model weights
            device: 'cuda', 'cpu', or 'auto'
            image_size: Input image size
        """
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = DeepfakeImageDetector(num_classes=2, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_size = image_size
        self.transform = self.model.get_input_transform(image_size)

        logger.info(f"Model loaded from {model_path}")

    def predict(self, image_path):
        """Predict on single image"""
        return self.model.predict_single(image_path, self.image_size)

    def predict_batch(self, image_paths):
        """Predict on multiple images"""
        return self.model.predict_batch(image_paths, self.image_size)
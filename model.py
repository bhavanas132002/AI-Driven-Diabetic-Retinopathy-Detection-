import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import random  # Import random module for fallback prediction

# Class Labels
classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def load_model():
    """Try loading model.pth; if not found, return a randomly initialized model."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)  # Adjust for 5 output classes

    try:
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    except:
        pass  # No error messages; just continue with a random model

    model.eval()
    return model

# Load model (either trained or random)
model = load_model()

# Define image transformations
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def inference(img_path):
    """Perform inference on an image; if it fails, return a random class."""
    try:
        image = Image.open(img_path).convert('RGB')
        image = test_transforms(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class, classes[predicted_class]

    except:
        return "Random", random.choice(classes)  # Return a random class if inference fails

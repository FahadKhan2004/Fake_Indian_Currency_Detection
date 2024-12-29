import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the model
def load_model(model_path, device):
    from torchvision.models import vit_b_16
    model = vit_b_16(weights=None)
    model.heads = torch.nn.Linear(in_features=768, out_features=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Define image preprocessing
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict class with probability
'''def predict(model, image_tensor, class_names, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, dim=1)
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input tensor device: {image_tensor.device}")
        return class_names[top_class], top_prob.item()
'''


def predict(model, image_tensor, class_names, device):
    model = model.to(device)  # Move model to device
    image_tensor = image_tensor.to(device)  # Move input tensor to device

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, dim=1)
        return class_names[top_class], top_prob.item()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
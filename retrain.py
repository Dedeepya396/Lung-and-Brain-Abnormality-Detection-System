import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms


# Retraining function
# Device configuration

# Define CNNClassifier class structure here if not imported from another file
class CNNClassifier(nn.Module):
    def init(self):
        super(CNNClassifier, self).init()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 13),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
def retrain_model(image_path, model_path, correct_label, learning_rate=0.001, epochs=5):
    # Load model and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    model.train()  # Set model to training mode

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    label_tensor = torch.tensor([correct_label]).to(device)

    # Set up criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Perform retraining
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(image_tensor)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()
    
    print(f"Retrained with label: {correct_label}")

    # Save the updated model
    torch.save(model.state_dict(), model_path)
    print(f"Model updated and saved to {model_path}")


from PIL import Image
import torch
import torch.nn as nn 
import torchvision.transforms as transforms

def GenerateOutput(model_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class CNNClassifier(nn.Module):
        def __init__(self):
            super(CNNClassifier, self).__init__()
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
    def predict(image_path, model, device):
        image = Image.open(image_path).convert('RGB') 
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        image = transform(image).unsqueeze(0).to(device) 

        model.eval() 
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

            class_names = ['Healthy Lungs', 'Pneumonia', 'Empyema', 'Pneumoperitoneum', 'Embolism', 'Fibrosis', 
                        'Metastases', 'Lymphadenopathy', 'Hypoplasia', 'Glioma', 'Healthy Brain', 'Meningioma', 'Pituitary']
            
            prediction = class_names[predicted.item()]
            print(f"The image is predicted to be: {prediction}")
            return prediction

    model = CNNClassifier().to(device) 
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.eval() 
    prediction = predict(image_path, model, device)
    return prediction

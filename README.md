# 🧠🫁 Lung and Brain Abnormality Detection

## 📌 Model Details
This project uses **Convolutional Neural Networks (CNNs)** built with **PyTorch** to detect abnormalities in lung and brain scans.

---

## 🧠🫁 Detected Classes

### 🫁 Lung:
- Healthy Lungs  
- Pneumonia  
- Empyema  
- Pneumoperitoneum  
- Embolism  
- Fibrosis  
- Metastases  
- Lymphadenopathy  
- Hypoplasia  

### 🧠 Brain:
- Healthy Brain  
- Glioma  
- Meningioma  
- Pituitary  

---

## ▶️ How to Run the App

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Run the application:
   ```bash
   python app.py
   ```

---

## 🧩 Dependencies

Make sure the following packages are installed:

- **PyTorch** – For model training and inference  
- **Pillow (PIL)** – For image handling  
- **Flask** – For the web interface  

Install all with:

```bash
pip install torch torchvision pillow flask
```

---

## 🧪 Test It Out

You can find test images in the `testdata/` folder.  
Use the web interface to upload and get predictions.

---

## ✅ Note

This project is for demonstration and educational purposes.  
Feel free to explore and modify as needed.

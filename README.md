# 🏠 House Price Prediction

### AI-Powered Property Valuation for Muzaffarpur, Bihar

A machine learning-based web application that estimates house/property prices in **Muzaffarpur, Bihar** using property characteristics such as locality, house size, number of bedrooms, total floors, and road-side status.

The prediction model was developed using **PyTorch in JupyterLab** and exported to **ONNX** for integration with the web application.

---

## 📌 About the Project

**House Price Prediction** is an interactive property valuation system that combines **Machine Learning and Web Development**.

Users can enter their property details through a simple web interface, and the trained model generates an estimated property value.

The application is designed around property characteristics that can influence house prices in different localities of Muzaffarpur.

### 🏡 Prediction Inputs

| Feature | Description |
|---|---|
| **Locality** | Selected locality/area category |
| **House Size** | Property size in square feet |
| **Bedrooms (BHK)** | Number of bedrooms |
| **Total Floors** | Number of floors in the property |
| **Road-Side Property** | Whether the property is located beside a road |

---

## ⚙️ How It Works

The project follows a complete machine learning workflow:

```text
        Property Dataset
              │
              ▼
      Data Preprocessing
              │
              ▼
       Feature Selection
              │
              ▼
       PyTorch Model
              │
              ▼
        Model Training
              │
              ▼
       Model Evaluation
              │
              ▼
        ONNX Export
              │
              ▼
       Web Application
              │
              ▼
    Estimated Property Value

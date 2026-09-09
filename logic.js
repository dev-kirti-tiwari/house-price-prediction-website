// Mean and Standard Deviation constants computed from your PyTorch preprocessing
const X_MEANS = [2.9995, 3201.2977, 3.9764, 1.6105, 0.6001];
const X_STDS  = [2.0001, 1616.6958, 1.3821, 0.7866, 0.4899];

// Target (Price_INR) normalization constants
const Y_MEAN = 38330436.48;
const Y_STD  = 52994323.20;

let session;

// Initialize ONNX Runtime session when page loads
async function initModel() {
    try {
        // Load the converted ONNX model file locally
        session = await ort.InferenceSession.create('./house_model.onnx');
        console.log("PyTorch Neural Network model loaded onto web runtime successfully.");
    } catch (e) {
        console.error("Failed to load ONNX model:", e);
    }
}

// Perform normalization calculation matching PyTorch Z-score conversion
function normalize(val, index) {
    return (val - X_MEANS[index]) / (X_STDS[index] + 1e-8);
}

async function runPrediction() {
    if (!session) {
        alert("Model is still initializing, please wait a moment.");
        return;
    }

    // Extract values directly from UI form controls
    const locality = parseFloat(document.getElementById('locality').value);
    const size = parseFloat(document.getElementById('size').value);
    const bedrooms = parseFloat(document.getElementById('bedrooms').value);
    const floors = parseFloat(document.getElementById('floors').value);
    const roadSide = parseFloat(document.getElementById('roadSide').value);

    // Apply normalization scaling across all 5 input features
    const inputsNormalized = [
        normalize(locality, 0),
        normalize(size, 1),
        normalize(bedrooms, 2),
        normalize(floors, 3),
        normalize(roadSide, 4)
    ];

    // Wrap array structure into an ONNX float32 tensor element
    const inputTensor = new ort.Tensor('float32', Float32Array.from(inputsNormalized), [1, 5]);

    try {
        // Feed the tensor matrix directly into the model layers
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        
        // Output from model is a normalized Z-score
        const predictedZScore = results.output.data[0];
        
        // Reverse Z-score normalization: Actual_Price = (ZScore * Std) + Mean
        const finalPriceINR = (predictedZScore * Y_STD) + Y_MEAN;

        // Render the calculated value nicely onto the screen
        const priceBox = document.getElementById('priceValue');
        priceBox.innerText = new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0
        }).format(Math.max(0, finalPriceINR)); // Ensure no display below ₹0

        document.getElementById('resultContainer').classList.remove('hidden');
    } catch (err) {
        console.error("Inference calculation error:", err);
    }
}

// Attach script execution triggers
document.addEventListener('DOMContentLoaded', initModel);
document.getElementById('predictBtn').addEventListener('click', runPrediction);
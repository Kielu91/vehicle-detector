from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
from detector import VehicleDetector

# Aplikacja
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detektor
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    
    # Znajdź model
    if os.path.exists('models/best.pt'):
        model_path = 'models/best.pt'
    else:
        model_path = 'yolov8n.pt'
    
    detector = VehicleDetector(model_path)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vehicle Detector</title>
        <style>
            body { 
                font-family: Arial; 
                margin: 0; 
                padding: 40px; 
                background: #f5f5f5; 
                display: flex;
                justify-content: center;
            }
            .container { 
                max-width: 600px; 
                background: white; 
                padding: 40px; 
                border-radius: 10px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                text-align: center;
            }
            .upload-area { 
                border: 3px dashed #007bff; 
                padding: 40px; 
                margin: 30px 0; 
                border-radius: 10px; 
                background: #f8f9ff;
            }
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
            }
            button { 
                background: #007bff;
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer;
                font-size: 16px;
            }
            button:hover { 
                background: #0056b3;
            }
            .result-image {
                max-width: 100%;
                margin-top: 30px;
                border-radius: 5px;
            }
            .loading {
                display: none;
                margin-top: 20px;
                color: #007bff;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Vehicle Detector</h1>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" id="imageFile" name="file" accept="image/*" required>
                    <br>
                    <button type="submit">Wykryj pojazdy</button>
                </div>
            </form>
            
            <div class="loading" id="loading">Analizuję obraz...</div>
            <div id="result"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                if (!fileInput.files.length) return;
                
                loading.style.display = 'block';
                result.innerHTML = '';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('confidence', '0.5');
                
                try {
                    const response = await fetch('/detect_image', {
                        method: 'POST',
                        body: formData
                    });
                    
                    loading.style.display = 'none';
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const imageUrl = URL.createObjectURL(blob);
                        result.innerHTML = `<img src="${imageUrl}" class="result-image">`;
                    } else {
                        result.innerHTML = '<p style="color: red;">Błąd</p>';
                    }
                } catch (error) {
                    loading.style.display = 'none';
                    result.innerHTML = '<p style="color: red;">Błąd połączenia</p>';
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Tylko obrazy")
    
    image_data = await file.read()
    result_image = detector.detect_and_draw(image_data, confidence_threshold=0.5)
    
    if result_image is None:
        raise HTTPException(status_code=500, detail="Błąd")
    
    return StreamingResponse(io.BytesIO(result_image), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
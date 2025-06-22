from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import io
import os
from detector import VehicleDetector

app = FastAPI(title="Vehicle Detector - Real AI", version="1.0.0")

# Detektor z prawdziwym YOLO
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    
    # Sprawdź dostępne modele
    model_options = [
        'models/best.pt',
        'yolov8n.pt'
    ]
    
    model_path = None
    for path in model_options:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        model_path = 'yolov8n.pt'
    
    print(f" Loading YOLO model: {model_path}")
    detector = VehicleDetector(model_path)
    print(" Real AI detector ready!")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vehicle Detector - Real AI</title>
        <style>
            body { 
                font-family: Arial; 
                margin: 0; 
                padding: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container { 
                max-width: 700px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1); 
                padding: 40px; 
                border-radius: 20px; 
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                text-align: center;
            }
            .upload-area { 
                border: 3px dashed rgba(255,255,255,0.5); 
                padding: 40px; 
                margin: 30px 0; 
                border-radius: 15px; 
                background: rgba(255,255,255,0.1);
                transition: all 0.3s ease;
            }
            .upload-area:hover { 
                border-color: #00f5ff; 
                background: rgba(0,245,255,0.1);
                transform: scale(1.02);
            }
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
                background: rgba(255,255,255,0.2);
                border: none;
                border-radius: 8px;
                color: white;
            }
            button { 
                background: linear-gradient(45deg, #00f5ff, #0066ff);
                color: white; 
                padding: 15px 40px; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer;
                font-size: 18px;
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,245,255,0.3);
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,245,255,0.5);
            }
            .result-image {
                max-width: 100%;
                margin-top: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            .loading {
                display: none;
                margin-top: 20px;
                font-size: 18px;
                color: #00f5ff;
            }
            .ai-badge {
                background: linear-gradient(45deg, #ff6b6b, #feca57);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
                display: inline-block;
                margin: 10px 0;
            }
            .tech-stack {
                font-size: 12px;
                opacity: 0.8;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Vehicle Detector</h1>
            <div class="ai-badge"> Real AI - YOLOv8 + PyTorch</div>
            <p>Upload obraz aby wykryć pojazdy przy użyciu prawdziwej sieci neuronowej!</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <p style="font-size: 18px; margin-bottom: 15px;"> Wybierz obraz z pojazdami</p>
                    <input type="file" id="imageFile" name="file" accept="image/*" required>
                    <br><button type="submit">🔍 Wykryj pojazdy (AI)</button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <p> AI analizuje obraz...</p>
            </div>
            
            <div id="result"></div>
            
            <div class="tech-stack">
                Tech Stack: PyTorch • YOLOv8 • FastAPI • Docker • Azure
            </div>
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
                
                try {
                    const response = await fetch('/detect_image', {
                        method: 'POST',
                        body: formData
                    });
                    
                    loading.style.display = 'none';
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const imageUrl = URL.createObjectURL(blob);
                        result.innerHTML = `
                            <h3>✅ AI Detection Complete:</h3>
                            <img src="${imageUrl}" class="result-image">
                            <p style="margin-top: 15px; font-size: 14px; opacity: 0.8;">
                                Wykryte przez YOLOv8 - różne kolory dla różnych typów pojazdów
                            </p>
                        `;
                    } else {
                        result.innerHTML = '<p style="color: #ff6b6b;">❌ Błąd AI detection</p>';
                    }
                } catch (error) {
                    loading.style.display = 'none';
                    result.innerHTML = '<p style="color: #ff6b6b;">❌ Błąd połączenia</p>';
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=500, detail="AI model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    try:
        image_data = await file.read()
        result_image = detector.detect_and_draw(image_data, confidence_threshold=0.5)
        
        if result_image is None:
            raise HTTPException(status_code=500, detail="AI detection failed")
        
        return StreamingResponse(
            io.BytesIO(result_image),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=ai_detected_{file.filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI detection error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_model": "YOLOv8" if detector else "not loaded",
        "model_path": detector.model_path if detector else None
    }

if __name__ == "__main__":
    import uvicorn
    print("🚗 Starting Vehicle Detector with Real AI...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
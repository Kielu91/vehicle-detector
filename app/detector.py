from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import torch

class VehicleDetector:
    """
    Klasa do wykrywania pojazdów przy użyciu modelu YOLO
    """ 
    
    def __init__(self, model_path='models/best.pt'):
        """
        Inicjalizuje detektor z wytrenowanym modelem
        
        Args:
            model_path (str): Ścieżka do pliku modelu .pt
        """
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
    
    def load_model(self):
        """
        Ładuje model YOLO
        """
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Model załadowany z: {self.model_path}")
            print(f"Używa device: {self.device}")
        except Exception as e:
            print(f"Błąd podczas ładowania modelu: {e}")
            # Fallback na pretrenowany model
            self.model = YOLO('yolov8n.pt')
            print("Używam pretrenowanego modelu YOLOv8n")
    
    def preprocess_image(self, image_data):
        """
        Przygotowuje obraz do detekcji
        
        Args:
            image_data (bytes): Dane obrazu w formacie bytes
            
        Returns:
            PIL.Image: Przetworzony obraz
        """
        try:
            # Konwertuj bytes na PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Konwertuj na RGB jeśli potrzeba
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"Błąd podczas przetwarzania obrazu: {e}")
            return None
    
    def detect_vehicles(self, image_data, confidence_threshold=0.5):
        """
        Wykrywa pojazdy na obrazie
        
        Args:
            image_data (bytes): Dane obrazu
            confidence_threshold (float): Próg pewności detekcji
            
        Returns:
            dict: Wyniki detekcji z bounding boxes i klasami
        """
        if self.model is None:
            return {"error": "Model nie został załadowany"}
        
        # Przetwórz obraz
        image = self.preprocess_image(image_data)
        if image is None:
            return {"error": "Nie można przetworzyć obrazu"}
        
        try:
            # Uruchom detekcję
            results = self.model(image, conf=confidence_threshold)
            
            # Przetwórz wyniki
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Współrzędne bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = self.model.names[class_id]
                        
                        detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": round(x1, 1),
                                "y1": round(y1, 1), 
                                "x2": round(x2, 1),
                                "y2": round(y2, 1)
                            }
                        })
            
            return {
                "detections": detections,
                "count": len(detections),
                "image_size": image.size,
                "model_used": self.model_path
            }
            
        except Exception as e:
            return {"error": f"Błąd podczas detekcji: {str(e)}"}
    
    def detect_and_draw(self, image_data, confidence_threshold=0.5):
        """
        Wykrywa pojazdy i zwraca obraz z narysowanymi bounding boxes
        
        Args:
            image_data (bytes): Dane obrazu
            confidence_threshold (float): Próg pewności
            
        Returns:
            bytes: Obraz z narysowanymi detekcjami
        """
        image = self.preprocess_image(image_data)
        if image is None:
            return None
        
        try:
            # Uruchom detekcję z parametrem save=False, show=False
            results = self.model(image, conf=confidence_threshold)
            
            # Narysuj wyniki na obrazie
            annotated_image = results[0].plot()
            
            # Konwertuj na PIL i zwróć jako bytes
            pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            
            # Zapisz do bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            return img_byte_arr.getvalue()
            
        except Exception as e:
            print(f"Błąd podczas rysowania detekcji: {e}")
            return None
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
        self.model = Nonefrom ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import torch
import os

class VehicleDetector:
    """
    Prawdziwy detektor pojazdów używający YOLOv8
    """
    
    def __init__(self, model_path='models/best.pt'):
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()
    
    def load_model(self):
        """
        Ładuje model YOLOv8
        """
        try:
            print(f"Loading YOLO model from: {self.model_path}")
            
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            else:
                print(f"Model {self.model_path} not found, using pretrained YOLOv8n")
                self.model = YOLO('yolov8n.pt')
            
            self.model.to(self.device)
            print(f" YOLO model loaded on device: {self.device}")
            
            # Sprawdź dostępne klasy
            vehicle_classes = []
            for class_id, class_name in self.model.names.items():
                if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    vehicle_classes.append(f"{class_id}: {class_name}")
            
            print(f" Vehicle classes: {', '.join(vehicle_classes)}")
            
        except Exception as e:
            print(f" Error loading YOLO model: {e}")
            raise
    
    def preprocess_image(self, image_data):
        """
        Przygotowuje obraz do detekcji
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def detect_and_draw(self, image_data, confidence_threshold=0.5):
        """
        Wykrywa pojazdy i rysuje ramki używając prawdziwego YOLO
        """
        image = self.preprocess_image(image_data)
        if image is None:
            return None
        
        try:
            print(f" Running YOLO detection with confidence {confidence_threshold}")
            
            # Uruchom detekcję YOLO
            results = self.model(image, conf=confidence_threshold, verbose=False)
            
            # Konwertuj PIL na OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            detection_count = 0
            
            # Przetwórz wyniki detekcji
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Pobierz dane z detekcji
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = self.model.names[class_id]
                        
                        # Filtruj tylko pojazdy
                        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
                        if class_name.lower() not in vehicle_classes:
                            continue
                        
                        detection_count += 1
                        
                        # Kolory dla różnych typów pojazdów
                        colors = {
                            'car': (0, 255, 0),        # Zielony
                            'truck': (255, 0, 0),      # Niebieski
                            'bus': (0, 165, 255),      # Pomarańczowy
                            'motorcycle': (255, 0, 255), # Magenta
                            'bicycle': (255, 255, 0)   # Cyan
                        }
                        color = colors.get(class_name.lower(), (0, 255, 0))
                        
                        # Narysuj prostokąt
                        cv2.rectangle(img_cv, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    color, 3)
                        
                        # Przygotuj tekst z etykietą
                        label = f"{class_name} {confidence:.2f}"
                        
                        # Parametry tekstu
                        font_scale = 0.8
                        thickness = 2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        
                        # Sprawdź rozmiar tekstu
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        
                        # Narysuj tło dla tekstu
                        cv2.rectangle(img_cv,
                                    (int(x1), int(y1) - text_height - baseline - 10),
                                    (int(x1) + text_width + 10, int(y1)),
                                    color, -1)
                        
                        # Napisz tekst
                        cv2.putText(img_cv, label,
                                  (int(x1) + 5, int(y1) - baseline - 5),
                                  font, font_scale, (255, 255, 255), thickness)
            
            print(f" YOLO detected {detection_count} vehicles")
            
            # Konwertuj z powrotem na PIL
            pil_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
            # Zapisz do bytes z wysoką jakością
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            return img_byte_arr.getvalue()
            
        except Exception as e:
            print(f" Error during YOLO detection: {e}")
            return None
    
    def get_model_info(self):
        """
        Zwraca informacje o modelu
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": self.model_path,
            "device": self.device,
            "model_type": "YOLOv8",
            "classes": self.model.names,
            "vehicle_classes": [name for name in self.model.names.values() 
                              if name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']]
        }
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
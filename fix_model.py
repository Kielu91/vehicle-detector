import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def fix_model_setup():
    """
    Naprawia setup modelu - zapewnia że model exists w właściwym miejscu
    """
    print("🔧 Naprawiam setup modelu...")
    
    # Utwórz folder models jeśli nie istnieje
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Folder models: {models_dir.absolute()}")
    
    # Sprawdź czy istnieje jakikolwiek model w folderze models
    model_files = list(models_dir.glob("*.pt"))
    if model_files:
        print(f"✅ Znaleziono modele: {[f.name for f in model_files]}")
        return True
    
    print("📥 Pobieram i kopiuję pretrenowany model YOLOv8...")
    
    try:
        # Pobierz pretrenowany model (jeśli jeszcze go nie ma)
        model = YOLO('yolov8n.pt')
        print("✅ Model YOLOv8n pobrany")
        
        # Sprawdź gdzie jest zapisany
        yolo_model_path = None
        possible_paths = [
            'yolov8n.pt',
            './yolov8n.pt',
            os.path.expanduser('~/.ultralytics/yolov8n.pt')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                yolo_model_path = path
                break
        
        if yolo_model_path:
            # Skopiuj do naszego folderu models
            target_path = models_dir / "best.pt"
            shutil.copy2(yolo_model_path, target_path)
            print(f"✅ Model skopiowany: {yolo_model_path} → {target_path}")
            
            # Sprawdź rozmiar pliku
            file_size = target_path.stat().st_size / (1024*1024)  # MB
            print(f"✅ Rozmiar modelu: {file_size:.1f} MB")
            
            return True
        else:
            print("❌ Nie znaleziono pliku yolov8n.pt")
            return False
            
    except Exception as e:
        print(f"❌ Błąd podczas kopiowania: {e}")
        return False

def test_model():
    """
    Testuje czy model działa
    """
    print("\n🧪 Testuję model...")
    
    try:
        # Sprawdź modele w folderze
        models_dir = Path("models")
        model_files = list(models_dir.glob("*.pt"))
        
        if model_files:
            model_path = model_files[0]
            print(f"📁 Używam: {model_path}")
        else:
            model_path = "yolov8n.pt"
            print(f"📁 Używam pretrenowanego: {model_path}")
        
        # Załaduj model
        model = YOLO(str(model_path))
        
        # Sprawdź klasy pojazdów
        vehicle_classes = []
        for class_id, class_name in model.names.items():
            if class_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                vehicle_classes.append(f"{class_id}: {class_name}")
        
        print("🚗 Klasy pojazdów w modelu:")
        for vc in vehicle_classes:
            print(f"   {vc}")
        
        print("✅ Model działa poprawnie!")
        return True
        
    except Exception as e:
        print(f"❌ Błąd testu: {e}")
        return False

def show_next_steps():
    """
    Pokazuje następne kroki
    """
    print("\n🚀 Następne kroki:")
    print("1. cd app")
    print("2. python main.py")
    print("3. Otwórz http://localhost:8000")
    print("4. Prześlij obraz z samochodami do testowania")
    print("\n💡 Pretrenowany YOLOv8 już wykrywa pojazdy bardzo dobrze!")

if __name__ == "__main__":
    print("🚗 === NAPRAWA SETUP MODELU ===\n")
    
    success = fix_model_setup()
    
    if success:
        test_model()
        show_next_steps()
    else:
        print("\n⚠️  Nie udało się skopiować modelu, ale to nie problem!")
        print("Aplikacja będzie działać z pretrenowanym modelem.")
        show_next_steps()
    
    print("\n✨ Setup zakończony!")
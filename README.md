# Ceronix Scratch Detection

A FastAPI-based application that uses **YOLOv8** for detecting scratches on **Ceronix displays**. The project provides an API interface for image-based inference, making it simple to detect and visualize surface defects.

---


## Introduction
The **Ceronix Scratch Detection Model** is built to identify scratches on Ceronix displays.  
It uses **YOLOv8**, trained with a custom dataset (`data.yaml`), and serves inference results via a **FastAPI** app.

---

## Features
- REST API powered by FastAPI.  
- YOLOv8 model trained on Ceronix scratch dataset.  
- Input image upload with detection visualization.  
- Sample test images included (`test_scratch_*.jpg`).  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mirudull-D/Ceronix_scratch.git
   cd Ceronix_scratch
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run FastAPI Server
```bash
uvicorn app:app --reload
```

Server will start at:  
👉 http://127.0.0.1:8000  

### Send an Image for Detection
Using `cURL`:
```bash
curl -X POST "http://127.0.0.1:8000/detect"      -F "file=@test_scratch_1.jpg"
```

Results will be saved in the `output/` directory.

---

## API Endpoints

| Method | Endpoint   | Description |
|--------|-----------|-------------|
| `POST` | `/detect` | Upload image and run YOLOv8 scratch detection |

---

## Dependencies
Main dependencies:
- [FastAPI](https://fastapi.tiangolo.com/)  
- [Uvicorn](https://www.uvicorn.org/)  
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)  
- [Pillow](https://pypi.org/project/Pillow/)  

---

## Configuration
- **Dataset Config**: Defined in `data.yaml`.  
- **Model Weights**: Custom YOLOv8 weights are stored under `runs/detect/train2/weights/`.  
- **Pretrained Model**: `yolov8n.pt` included for transfer learning or fallback.  
- **Output Directory**: Processed images and results are saved in `output/`.  
- **Port**: Default is `8000`. Change with:
  ```bash
  uvicorn app:app --host 0.0.0.0 --port 9000
  ```

---

## Examples
- Input: `test_scratch_1.jpg`  
- Output: Detected scratch image in `output/`  

---

## Troubleshooting
- **Error:** `ModuleNotFoundError: No module named 'ultralytics'`  
  ✅ Run `pip install ultralytics`.  

- **Error:** Server not starting with `uvicorn`.  
  ✅ Install with `pip install uvicorn`.  

- **Error:** Detection not saving outputs.  
  ✅ Ensure `output/` directory has write permissions.  

---

## Contributors
- [Mirudull D](https://github.com/Mirudull-D)  
- [Raghav S](https://github.com/raghav-404)  

---

## License
This project is licensed under the **MIT License**.  

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from pathlib import Path
import shutil, uuid, os, threading, time

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once
model1 = YOLO("runs/detect/train2/weights/best.pt")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount output folder so React can access result images
app.mount("/output", StaticFiles(directory="output"), name="output")


def schedule_cleanup(path: Path, delay: int = 60):
    """Deletes files/folders after delay seconds (to avoid storage bloat)."""
    def _delete():
        time.sleep(delay)
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    threading.Thread(target=_delete, daemon=True).start()


async def run_yolo(model: YOLO, file: UploadFile):
    """Run YOLO inference and return image + detections."""
    run_name = f"run_{uuid.uuid4().hex[:8]}"
    input_path = UPLOAD_DIR / f"{run_name}_{file.filename}"

    # Save uploaded file
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    results = model.predict(
        source=str(input_path),
        save=True,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=True
    )

    save_dir = Path(results[0].save_dir)
    output_files = list(save_dir.glob("*.jpg"))
    if not output_files:
        return JSONResponse({"error": "No output image generated"}, status_code=500)

    output_image_path = output_files[0]
    output_url = f"/output/{run_name}/{output_image_path.name}"

    # Extract class labels
    scratches = [
        results[0].names[int(c)]
        for c in results[0].boxes.cls.cpu().numpy()
    ] if results[0].boxes else []

    # Cleanup after delay
    schedule_cleanup(input_path, delay=60)
    schedule_cleanup(save_dir, delay=60)

    return {"output_url": output_url, "scratches": scratches}


@app.post("/predict2")
async def predict1(file: UploadFile = File(...)):
    return await run_yolo(model1, file)

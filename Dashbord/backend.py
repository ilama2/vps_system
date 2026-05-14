import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from openai import OpenAI



ACEG_DIR = Path("/Users/lama/Desktop/realtime_Acg")

CONFIG = Path(
    "/Users/lama/Desktop/realtime_Acg/my_outputs 3/2026-05-02_18-49-27-423_workspace_dataset_DINOv2Encoder_TransformerHead_map.yaml"
)

LIVE_DIR = ACEG_DIR / "realtime_live"
RGB_DIR = LIVE_DIR / "rgb"
CALIB_DIR = LIVE_DIR / "calibration"
OUTPUT_DIR = ACEG_DIR / "outputs"

RGB_DIR.mkdir(parents=True, exist_ok=True)
CALIB_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_PATH = RGB_DIR / "frame_000000.jpg"
CALIB_PATH = CALIB_DIR / "frame_000000.txt"

PYTHON = sys.executable

MAX_EXPECTED_CONF = 500


client = OpenAI()



app = FastAPI(title="ACE-G Image Localization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PoseExplainRequest(BaseModel):
    confidence_percent: float
    confidence_raw: float | None = None
    translation: dict
    quaternion: dict | None = None
    orientation: dict
    position: dict
    scene_status: str




def clean_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def write_calibration(image_path):
    img = Image.open(image_path)
    width, height = img.size

    fx = 1000
    fy = 1000
    cx = width / 2
    cy = height / 2

    CALIB_PATH.write_text(
        f"{fx} 0 {cx}\n"
        f"0 {fy} {cy}\n"
        "0 0 1\n"
    )


def clean_folder(folder: Path):
    for f in folder.glob("*"):
        if f.is_file():
            f.unlink()


def find_latest_pose_file():
    pose_files = list(OUTPUT_DIR.rglob("*registered_poses.txt"))
    pose_files += list(OUTPUT_DIR.rglob("registered_poses.txt"))

    if not pose_files:
        return None

    return max(pose_files, key=lambda p: p.stat().st_mtime)


@app.get("/")
def root():
    return {
        "message": "ACE-G backend is running",
        "aceg_dir": str(ACEG_DIR),
        "config": str(CONFIG),
    }


@app.post("/explain_pose")
def explain_pose(data: PoseExplainRequest):
    try:
        prompt = f"""
You are explaining the output of an ACE-G visual relocalization system.

The system estimates the camera pose of a query image inside a mapped environment.

Explain the result clearly for a user who is not an expert in computer vision.

Use simple, short language.

Explain:
1. What direction the camera is facing.
2. Where the camera is located relative to the map origin.
3. How reliable the localization is.
4. Whether the image likely belongs to the mapped scene.
5. What the user should do if confidence is low.

Do not mention raw JSON.
Do not overclaim.
Do not say the result is perfect.
Keep the answer between 3 and 5 sentences.

Data:
Confidence percent: {data.confidence_percent}%
Raw confidence: {data.confidence_raw}
Scene status: {data.scene_status}

Camera position interpretation:
X: {data.position.get("x")}
Y: {data.position.get("y")}
Z: {data.position.get("z")}

Camera orientation interpretation:
Horizontal: {data.orientation.get("horizontal")}
Vertical: {data.orientation.get("vertical")}
Roll: {data.orientation.get("rollText")}

Raw translation:
{data.translation}

Raw quaternion:
{data.quaternion}
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You explain visual localization and camera pose results clearly and accurately.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
        )

        explanation = response.choices[0].message.content

        diag_prompt = f"""
        Generate 3 short diagnostic bullet points for this ACE-G localization result.

        Rules:
        - Short and technical.
        - Mention localization quality.
        - Mention scene overlap.
        - Mention how to improve localization if needed.
        - Return ONLY bullet points.
        """

        diag_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You generate concise diagnostics for visual localization systems."
                },
                {
                    "role": "user",
                    "content": diag_prompt + "\n\n" + prompt
                },
            ],
            temperature=0.3,
        )

        diag_text = diag_response.choices[0].message.content

        diagnostics = []

        for line in diag_text.splitlines():
            line = line.strip("-• ").strip()

            if line:
                diagnostics.append(line)

        return JSONResponse(
            {
                "success": True,
                "explanation": explanation,
                "diagnostics": diagnostics,
            }
        )

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "explanation": "AI explanation could not be generated.",
                "error": str(e),
            },
            status_code=500,
        )


@app.post("/localize")
async def localize_image(file: UploadFile = File(...)):
    start = time.time()

    try:
        clean_folder(RGB_DIR)
        clean_folder(CALIB_DIR)

        with FRAME_PATH.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        write_calibration(FRAME_PATH)

        cmd = [
            PYTHON,
            "-m",
            "ace_g.register_images",
            "--config",
            str(CONFIG),
            "--device",
            "cpu",
            "--num_data_workers",
            "0",
            "--max_estimates",
            "1",
            "--hypotheses",
            "16",
            "--threshold",
            "20",
            "--dataset.rgb_files",
            str(RGB_DIR / "*.jpg"),
            "--dataset.calibration_files",
            str(CALIB_DIR / "*.txt"),
            "--dataset.calibration_source",
            "dataset",
            "--output_dir",
            str(OUTPUT_DIR),
        ]

        env = {
            **os.environ,
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "PYTHONPATH": f"{ACEG_DIR}/src:{ACEG_DIR}/dsacstar",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }

        result = subprocess.run(
            cmd,
            cwd=str(ACEG_DIR),
            env=env,
            text=True,
            capture_output=True,
        )

        elapsed = time.time() - start
        pose_file = find_latest_pose_file()

        if pose_file and pose_file.exists():
            raw_pose = pose_file.read_text().strip()
            values = raw_pose.split()

            if len(values) >= 18:
                confidence = float(values[-1])

                pose_text = {
                    "image": values[0],
                    "quaternion": {
                        "qw": round(float(values[1]), 6),
                        "qx": round(float(values[2]), 6),
                        "qy": round(float(values[3]), 6),
                        "qz": round(float(values[4]), 6),
                    },
                    "translation": {
                        "tx": round(float(values[5]), 6),
                        "ty": round(float(values[6]), 6),
                        "tz": round(float(values[7]), 6),
                    },
                    "confidence_raw": round(confidence),
                    "confidence_percent": min(
                        round((confidence / MAX_EXPECTED_CONF) * 100),
                        100,
                    ),
                }
            else:
                pose_text = {
                    "raw": raw_pose,
                }

            pose_file_path = str(pose_file)

        else:
            pose_text = None
            pose_file_path = None

        return JSONResponse(
            {
                "success": result.returncode == 0,
                "inference_time": round(elapsed, 2),
                "image_path": str(FRAME_PATH),
                "calibration_path": str(CALIB_PATH),
                "pose_file": pose_file_path,
                "pose": pose_text,
                "stdout": "",
                "stderr": "",
            }
        )

    except Exception as e:
        elapsed = time.time() - start

        return JSONResponse(
            {
                "success": False,
                "inference_time": round(elapsed, 2),
                "pose": None,
                "stdout": "",
                "stderr": str(e),
            },
            status_code=500,
        )

import os
import time
import base64
import openai
import requests
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from astroquery.simbad import Simbad

# Load API keys
load_dotenv()
ASTROMETRY_API_KEY = os.getenv("ASTROMETRY_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
#openai.base_url = "https://openrouter.ai/api/v1"

# there is way too many debugging statements here, clean if you have time

#print(f"--- DEBUG: OPENROUTER_API_KEY loaded value: '{OPENROUTER_KEY}' ---")
if OPENROUTER_KEY and len(OPENROUTER_KEY) < 10:
    print("FATAL: Key loaded but appears too short. Check value.")

if OPENROUTER_KEY:
    print("DEBUG: OpenRouter Key successfully read. (Length:", len(OPENROUTER_KEY), ")")
else:
    print("FATAL: OPENROUTER_API_KEY is NOT set or is empty in Python environment.")


try:
    astro = openai.OpenAI(
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
except Exception as e:
    # Handle case where API key might be missing
    print(f"Error initializing OpenAI client: {e}")
    astro = None # Set to None to prevent further errors

app = FastAPI()

# === Serve frontend ===
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/CosmicQuery.html")

# === Models ===
class AnalyzeRequest(BaseModel):
    base64Image: str
    mimeType: str

class ChatRequest(BaseModel):
    message: str

# === Helpers ===


ASTROMETRY_API_URL = "https://nova.astrometry.net/api"

def submit_to_astrometry(image_bytes):
    print("Logging in to Astrometry.net...")
    login_url = f"{ASTROMETRY_API_URL}/login"
    login_payload = {"apikey": ASTROMETRY_API_KEY}
    login_data_payload = {'request-json': json.dumps(login_payload)}
    s = requests.Session()

    # 1. Login using the session object
    login_resp = s.post(login_url, data=login_data_payload, allow_redirects=True)

    if login_resp.status_code != 200:
        error_text = login_resp.text[:100].replace('\n', ' ')
        raise Exception(f"Astrometry Login Failed: HTTP Status {login_resp.status_code}. Response: {error_text}...")

    try:
        login_data = login_resp.json()
    except requests.exceptions.JSONDecodeError:
        raise Exception("Astrometry.net returned non-JSON data during login. Check API key.")

    if "session" not in login_data:
        raise Exception(f"Login failed: {login_data}")

    session_id = login_data["session"]
    print(f"✅ Logged in with session: {session_id}")

    # --- 2. Upload the image (using the session object) ---
    upload_url = f"{ASTROMETRY_API_URL}/upload"
    upload_payload = {
        "session": session_id,
        "publicly_visible": "y",
        "allow_commercial_use": "d",
        "allow_modifications": "d"
    }

    print("Submitting image to Astrometry.net...")
    files = {'file': ('image.jpg', image_bytes, 'application/octet-stream')}
    data = {'request-json': json.dumps(upload_payload)}
    # Use the session object for the upload
    upload_resp = s.post(upload_url, data=data, files=files)

    if upload_resp.status_code != 200:
        raise Exception(f"Upload HTTP error {upload_resp.status_code}: {upload_resp.text}")

    try:
        upload_data = upload_resp.json()
    except Exception:
        raise Exception(f"Upload failed: non-JSON response:\n{upload_resp.text}")

    if "subid" not in upload_data:
        raise Exception(f"Astrometry upload failed: {upload_data}")

    subid = upload_data["subid"]
    print(f"Submission ID: {subid}. Waiting for Astrometry.net to solve image...")

    # --- 3. Poll submission for job ID (using the session object) ---
    job_id = None
    for _ in range(45):
        # Use the session object for polling
        status_resp = s.get(f"{ASTROMETRY_API_URL}/submissions/{subid}", allow_redirects=True)

        if status_resp.status_code != 200:
             raise Exception(f"Submission status check failed: HTTP {status_resp.status_code}")

        status_data = status_resp.json()

        jobs = status_data.get("jobs", [])
        if jobs and jobs[0]:
            job_id = jobs[0]
            break
        time.sleep(4)

    if not job_id:
        raise Exception("Timeout waiting for Astrometry.net to assign a job ID.")

    print(f"Got job ID: {job_id}")

    # --- 4. Poll job status until solved (using the session object) ---
    for i in range(60):
      #  job_status_url = f"{ASTROMETRY_API_URL}/jobs/{job_id}/status"
      #  job_status_resp = s.get(job_status_url, allow_redirects=True, verify=False)

        job_info_url = f"{ASTROMETRY_API_URL}/jobs/{job_id}/info?session={session_id}"
        job_info_resp = s.get(job_info_url, allow_redirects=True, verify=False)
        if job_info_resp.status_code != 200:
            raise Exception(f"Job status check failed: HTTP {job_info_resp.status_code}")

        try:
            info_data = job_info_resp.json()
            job_status = info_data.get("status")
        except requests.exceptions.JSONDecodeError:
            # If JSON decoding fails, we treat it as a temporary redirection/pending status
            print(f"Job status check returned non-JSON data (attempt {i+1}). Waiting...")
            job_status = "pending"

        print(f"Job {job_id} status: {job_status}")

        if job_status == "success":
            print(f"✅ Astrometry job {job_id} solved successfully.")
            return job_id
        elif job_status == "failure":
            raise Exception("Astrometry job failed to solve.")
        time.sleep(4)

    raise Exception("Astrometry.net job did not finish in time.")

def get_astrometry_results(job_id):
    base_url = "https://nova.astrometry.net"

    # Info endpoint → valid JSON
    info_resp = requests.get(f"{base_url}/api/jobs/{job_id}/info")
    try:
        info = info_resp.json()
    except ValueError:
        print("⚠️ Warning: Could not parse /info response:", info_resp.text[:200])
        info = {}

    objects = info.get("objects_in_field", [])
    ra = info.get("calibration", {}).get("ra")
    dec = info.get("calibration", {}).get("dec")

    # Image URL (don’t fetch or parse)
    annotated_url = f"{base_url}/annotated_display/{job_id}"

    return {
        "ra": ra,
        "dec": dec,
        "objects": objects,
        "annotated_url": annotated_url
    }

def query_object_details(object_name):
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('otype', 'distance', 'flux(V)', 'coordinates')

    try:
        result = custom_simbad.query_object(object_name)
    except Exception:
        return None

    if result:
        return {
            "name": object_name,
            "type": result['OTYPE'][0] if 'OTYPE' in result.colnames else "Unknown",
            "distance": f"{result['Distance_distance'][0]} pc" if 'Distance_distance' in result.colnames and result['Distance_distance'][0] else "Unknown",
            "magnitude": str(result['FLUX_V'][0]) if 'FLUX_V' in result.colnames and result['FLUX_V'][0] else "Unknown",
            "coordinates": result['COORDINATES'][0] if 'COORDINATES' in result.colnames else "Unknown"
        }
    return None


@app.post("/api/analyze")
async def analyze_image(data: AnalyzeRequest):
    # === OUTER TRY BLOCK (covers all logic) ===
    try:
        # Decode user image
        image_bytes = base64.b64decode(data.base64Image)

        # Submit and solve
        job_id = submit_to_astrometry(image_bytes)
        astro_data = get_astrometry_results(job_id)

        # Build textual context
        if astro_data["objects"]:
            object_name = astro_data["objects"][0]
            simbad_data = query_object_details(object_name)
            if simbad_data:
                description = (
                    f"Object: {simbad_data['name']}\n"
                    f"Type: {simbad_data['type']}\n"
                    f"Distance: {simbad_data['distance']}\n"
                    f"Apparent Magnitude: {simbad_data['magnitude']}\n"
                    f"Coordinates: {simbad_data['coordinates']}"
                )
            else:
                description = f"Detected object: {object_name}, but no additional SIMBAD data found."
        else:
            description = "Astrometry.net could not identify specific named objects."

        # === LLM call ===
        base64_image_data_url = f"data:{data.mimeType};base64,{data.base64Image}"

        # --- INNER TRY BLOCK (for LLM safety) ---
        try:
            gpt_response = astro.chat.completions.create(
                model="google/gemini-2.0-flash-exp:free",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional astronomical image analyzer that helps inquisitive minds understand more about the cosmos."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Analyze the image and the following data:\n{description}\n\nPlease tell the user what is in this image and its astronomical significance. Make sure every idea presented is of utmost scientific accuracy. Additionally, do not use typical formatting, talk in a very human, connective way. Be mythodical when you describe cold hard facts, and kind when asked questions. Do not say more than nessessary, only what the facts are, then talk more if the user wishes it."},
                            {"type": "image_url", "image_url": {"url": base64_image_data_url}}
                        ]
                    }
                ],
                max_tokens=700
            )
            # CRITICAL FIX: The check and assignment MUST be inside the try block where gpt_response is created
            if isinstance(gpt_response, str):
                raise Exception(f"LLM API returned a raw string error: {gpt_response[:100]}")

            summary = gpt_response.choices[0].message.content

        except Exception as llm_e:
            # Handle LLM failure gracefully, but allow outer block to catch and log
            print(f"LLM Processing Error: {llm_e}")
            summary = "The AI guide is currently offline. We successfully identified the objects, but cannot provide a description. This is due to the free model being rate-limited upsteam. Apologies!"
        # --- END INNER TRY BLOCK ---


        # === FINAL RETURN ===
        return {
            "reply": summary,
            "annotatedImage": astro_data["annotated_url"]
        }

    # === OUTER EXCEPT BLOCK (catches Astrometry, Decoding, and general errors) ===
    except Exception as e:
        print("Error:", e)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(data: ChatRequest):
    try:
        response = astro.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[
                {"role": "system", "content": "You are a professional astronomical image analyzer that helps inquisitive minds understand more about the cosmos."},
                {"role": "user", "content": data.message}
            ],
            max_tokens=500
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        print("Chat error:", e)
        raise HTTPException(status_code=500, detail=str(e))

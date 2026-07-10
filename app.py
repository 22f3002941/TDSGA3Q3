from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dateutil.parser import parse
import ollama
import json
from pydantic import BaseModel, create_model, Field, ConfigDict
from typing import Optional, Dict, Any

from fastapi.responses import JSONResponse
import base64
import re

from typing import List
from ollama import chat

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[float] = None
    tax: Optional[float] = None
    currency: Optional[str] = None

SYSTEM_PROMPT = """
You are an expert invoice information extraction system.

Extract information from invoice text.

Return ONLY valid JSON matching this schema:
{
  "invoice_no": string | null,
  "date": string | null,
  "vendor": string | null,
  "amount": number | null,
  "tax": number | null,
  "currency": string | null
}

Rules:
- invoice_no: copy exactly.
- date: return ISO format YYYY-MM-DD if possible.
- vendor: the issuing company, not the client.
- amount: subtotal before tax only.
- tax: tax amount only.
- currency: INR, USD, EUR, GBP, etc.
- Return null if a field cannot be found.
"""

@app.post("/extract", response_model=InvoiceResponse)
def extract(req: InvoiceRequest):
    schema = InvoiceResponse.model_json_schema()

    response = ollama.chat(
        model="gemma3:latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.invoice_text}
        ],
        format=schema,
        options={
            "temperature": 0,
            "top_p": 0.1,
            "top_k": 10,
            "repeat_penalty": 1.0,
            "num_predict": 256
        }
    )

    content = response["message"]["content"]

    try:
        data = json.loads(content)
    except Exception:
        data = {}

    result = {
        "invoice_no": data.get("invoice_no"),
        "date": data.get("date"),
        "vendor": data.get("vendor"),
        "amount": data.get("amount"),
        "tax": data.get("tax"),
        "currency": data.get("currency"),
    }

    if result["date"]:
        try:
            result["date"] = parse(result["date"]).date().isoformat()
        except Exception:
            result["date"] = None

    return result

# Question 4

class DynamicExtractRequest(BaseModel):
    text: str
    schema: Dict[str, str]

TYPE_MAP = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "float": float,
    "number": float,
    "date": str,
    "boolean": bool,
    "bool": bool,
}

def build_dynamic_model(schema_dict: Dict[str, str]):
    fields = {}
    for key, type_name in schema_dict.items():
        py_type = TYPE_MAP.get(str(type_name).lower(), str)
        fields[key] = (Optional[py_type], Field(default=None))
    DynamicModel = create_model(
        "DynamicExtractResponse",
        __config__=ConfigDict(extra="forbid"),
        **fields
    )
    return DynamicModel

def normalize_value(value: Any, type_name: str):
    if value is None:
        return None

    t = str(type_name).lower()

    if t == "date":
        try:
            return parse(str(value)).date().isoformat()
        except Exception:
            return None

    if t in ("integer", "int"):
        try:
            return int(value)
        except Exception:
            return None

    if t in ("float", "number"):
        try:
            return float(value)
        except Exception:
            return None

    if t in ("boolean", "bool"):
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
        return None

    return str(value)

@app.get("/")
def home():
    return {"status": "All APIs running"}

@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicExtractRequest):
    if not req.schema:
        raise HTTPException(status_code=400, detail="schema cannot be empty")

    DynamicModel = build_dynamic_model(req.schema)
    json_schema = DynamicModel.model_json_schema()

    system_prompt = (
        "You are a precise information extraction system.\n"
        "Return ONLY valid JSON that matches the given schema exactly.\n"
        "Do not add extra keys. Use null for missing values.\n"
        "If a field is typed as date, return it in YYYY-MM-DD format.\n"
    )

    user_prompt = f"""
Text:
{req.text}

Requested schema:
{json.dumps(req.schema, indent=2)}

Return JSON only.
"""

    response = ollama.chat(
        model="gemma3:latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=json_schema,
        options={
            "temperature": 0,
            "top_p": 0.1,
            "top_k": 10,
            "repeat_penalty": 1.0,
            "num_predict": 256,
        },
    )

    content = response["message"]["content"]

    try:
        data = json.loads(content)
    except Exception:
        data = {}

    result = {}
    for key, type_name in req.schema.items():
        result[key] = normalize_value(data.get(key, None), type_name)

    return result

# Question 6
"""
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def empty_response() -> Dict[str, Any]:
    return {
        "rows": 0,
        "columns": [],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": []
    }

@app.post("/analyze")
def analyze(req: AudioRequest):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    import io
    import wave
    import numpy as np

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(n_frames)

        if sampwidth == 1:
            dtype = np.uint8
        elif sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            raise HTTPException(status_code=400, detail="Unsupported WAV sample width")

        data = np.frombuffer(frames, dtype=dtype)

        if n_channels > 1:
            data = data.reshape(-1, n_channels)[:, 0]

        if data.size == 0:
            raise HTTPException(status_code=400, detail="Empty audio payload")

        result = {
            "rows": int(data.size),
            "columns": ["점수"],
            "mean": {"점수": float(np.mean(data))},
            "std": {"점수": float(np.std(data))},
            "variance": {"점수": float(np.var(data))},
            "min": {"점수": float(np.min(data))},
            "max": {"점수": float(np.max(data))},
            "median": {"점수": float(np.median(data))},
            "mode": {"점수": float(np.bincount(data.astype(np.int64) - data.min()).argmax() + data.min())},
            "range": {"점수": float(np.max(data) - np.min(data))},
            "allowed_values": {},
            "value_range": {"점수": [float(np.min(data)), float(np.max(data))]},
            "correlation": []
        }
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid WAV audio")

@app.get("/")
def home():
    return {"status": "ok"}

"""
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import json
import csv
import io
import statistics

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def empty_response():
    return {
        "rows": 0,
        "columns": [],
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": []
    }

def compute_stats(vals):
    vals = [float(v) for v in vals if v is not None and str(v).strip() != ""]
    if not vals:
        return None
    modes = statistics.multimode(vals)
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "variance": statistics.variance(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "median": statistics.median(vals),
        "mode": modes[0] if modes else None,
        "range": max(vals) - min(vals),
        "value_range": [min(vals), max(vals)]
    }

@app.post("/analyze")
def analyze(req: AudioRequest):
    try:
        raw = base64.b64decode(req.audio_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    text = None
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc).strip()
            break
        except Exception:
            pass

    if text is None:
        raise HTTPException(status_code=400, detail="Payload is not text")

    rows = None

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = [obj]
    except Exception:
        pass

    if rows is None:
        try:
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
        except Exception:
            rows = None

    if rows is None or len(rows) == 0:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            rows = [{"점수": x} for x in lines]

    if rows is None or len(rows) == 0:
        return JSONResponse(content=empty_response())

    columns = list(rows[0].keys())
    result = empty_response()
    result["rows"] = len(rows)
    result["columns"] = columns

    for col in columns:
        vals = [r.get(col) for r in rows]
        stats = compute_stats(vals)
        if stats:
            result["mean"][col] = stats["mean"]
            result["std"][col] = stats["std"]
            result["variance"][col] = stats["variance"]
            result["min"][col] = stats["min"]
            result["max"][col] = stats["max"]
            result["median"][col] = stats["median"]
            result["mode"][col] = stats["mode"]
            result["range"][col] = stats["range"]
            result["value_range"][col] = stats["value_range"]

    return JSONResponse(content=result)

#Question 7
from dateparser import parse

class LineItem(BaseModel):
    sku: str
    quantity: int
    unit_price: int


class Invoice(BaseModel):
    vendor: str
    currency: str
    total_amount: int
    invoice_date: str
    due_in_days: int
    is_paid: bool
    priority: str
    contact_email: str
    line_items: List[LineItem]
    item_count: int


class Request(BaseModel):
    document_id: str
    text: str
    schema: dict


SYSTEM_PROMPT = """
You are an invoice extraction engine.

Return ONLY a valid JSON object.

Requirements:
- vendor: exactly as written.
- currency: one of USD, EUR, GBP, INR, JPY.
- total_amount: integer only.
- invoice_date: always YYYY-MM-DD.
- Convert month names from any language to YYYY-MM-DD.
- is_paid: boolean.
- priority: exactly one of low, normal, high, urgent.
- contact_email: lowercase.
- line_items: preserve order.
- item_count: number of line_items.
- due_in_days:Return the number of days until payment is due.
- contact_email:
    Copy the email address EXACTLY from the document.
    Do NOT correct spelling.
    Do NOT expand domains.
    Only convert to lowercase.

Examples for due_in_days:
Net 30 -> 30
Net 45 -> 45
Payable within 21 days -> 21
Due in three weeks -> 21
Due in two weeks -> 14
Due in one week -> 7
Due in 60 days -> 60

Return only the integer.

Do not output markdown or explanations."""

def extract_email(text):
    m = EMAIL_RE.search(text)
    if m:
        return m.group(0).lower()
    return None

def normalize_date(value):
    if not value:
        return value

    dt = parse(value)

    if dt:
        return dt.strftime("%Y-%m-%d")

    return value

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

@app.post("/extract1", response_model=Invoice)
def extract1(req: Request):

    response = chat(
        model="gemma3:latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.text}
        ],
        #format=req.schema,
        options={
            "temperature": 0
        }
    )

    data = json.loads(response.message.content)

    email = extract_email(req.text)
    if email:
        data["contact_email"] = email

    data["invoice_date"] = normalize_date(data["invoice_date"])
    data["contact_email"] = data["contact_email"].lower()
    data["priority"] = data["priority"].lower()
    data["item_count"] = len(data["line_items"])

    return Invoice(**data)
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from functools import reduce
import math
import os
import openai
import requests
from dotenv import load_dotenv

load_dotenv()

OFFICIAL_EMAIL = os.getenv("OFFICIAL_EMAIL", "YOUR CHITKARA EMAIL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

app = FastAPI(title="BFHL API")

# Allow public access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Small request size guard (reject very large payloads)
MAX_CONTENT_LENGTH = 10 * 1024  # 10 KB


@app.middleware("http")
async def limit_content_length(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_CONTENT_LENGTH:
        return JSONResponse(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content={"is_success": False, "official_email": OFFICIAL_EMAIL, "error": "Payload too large"})
    return await call_next(request)


class BFHLRequest(BaseModel):
    fibonacci: Optional[int] = None
    prime: Optional[List[int]] = None
    lcm: Optional[List[int]] = None
    hcf: Optional[List[int]] = None
    AI: Optional[str] = None


def make_success(data):
    return {"is_success": True, "official_email": OFFICIAL_EMAIL, "data": data}


def make_error(message):
    return {"is_success": False, "official_email": OFFICIAL_EMAIL, "error": message}


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r = int(math.sqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def lcm_two(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def lcm_list(nums: List[int]) -> int:
    return reduce(lcm_two, nums)


def hcf_list(nums: List[int]) -> int:
    return reduce(gcd, nums)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"is_success": False, "official_email": OFFICIAL_EMAIL, "error": exc.detail})


@app.post("/bfhl")
async def bfhl(req: BFHLRequest):
    # Ensure exactly one key is present
    provided = {k: v for k, v in req.dict().items() if v is not None}
    if len(provided) != 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request must contain exactly one of: fibonacci, prime, lcm, hcf, AI")

    key = list(provided.keys())[0]
    val = provided[key]

    try:
        if key == "fibonacci":
            n = val
            if not isinstance(n, int) or n < 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="fibonacci must be a non-negative integer")
            if n > 1000:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="fibonacci n too large")
            res = []
            a, b = 0, 1
            for _ in range(n):
                res.append(a)
                a, b = b, a + b
            return JSONResponse(status_code=200, content=make_success(res))

        if key == "prime":
            arr = val
            if not isinstance(arr, list) or not all(isinstance(x, int) for x in arr):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="prime must be an array of integers")
            res = [x for x in arr if is_prime(x)]
            return JSONResponse(status_code=200, content=make_success(res))

        if key == "lcm":
            arr = val
            if not isinstance(arr, list) or len(arr) == 0 or not all(isinstance(x, int) for x in arr):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="lcm must be a non-empty array of integers")
            res = lcm_list(arr)
            return JSONResponse(status_code=200, content=make_success(res))

        if key == "hcf":
            arr = val
            if not isinstance(arr, list) or len(arr) == 0 or not all(isinstance(x, int) for x in arr):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="hcf must be a non-empty array of integers")
            res = hcf_list(arr)
            return JSONResponse(status_code=200, content=make_success(res))

        if key == "AI":
            question = val
            if not isinstance(question, str) or len(question.strip()) == 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="AI must be a non-empty string question")

            # Dispatch to provider
            def call_openai(q: str) -> str:
                if not OPENAI_API_KEY:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="OpenAI API key not configured")
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Answer the user's question concisely in a single word (one word only). If a one-word answer is impossible, return the most relevant single-word concept."},
                        {"role": "user", "content": q}
                    ],
                    max_tokens=10,
                    temperature=0.0,
                )
                text = resp.choices[0].message.content.strip()
                return text.split()[0]

            def call_anthropic(q: str) -> str:
                # Uses Anthropic's REST complete endpoint. Requires ANTHROPIC_API_KEY env var.
                if not ANTHROPIC_API_KEY:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Anthropic API key not configured")
                url = "https://api.anthropic.com/v1/complete"
                prompt = f"\nHuman: {q}\nAssistant:"
                headers = {
                    "x-api-key": ANTHROPIC_API_KEY,
                    "Content-Type": "application/json",
                }
                body = {
                    "model": "claude-2.1",
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.0,
                }
                try:
                    r = requests.post(url, json=body, headers=headers, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    text = data.get("completion", "").strip()
                    return text.split()[0] if text else ""
                except requests.RequestException as e:
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Anthropic error: {str(e)}")

            def call_gemini(q: str) -> str:
                # Call Google Generative Language (Gemini) REST endpoint using API key
                if not GEMINI_API_KEY:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Gemini API key not configured")
                # Use the public REST endpoint for text generation (model name may vary)
                url = f"https://generativelanguage.googleapis.com/v1/models/text-bison-001:generate?key={GEMINI_API_KEY}"
                body = {
                    "prompt": {"text": q},
                    "maxOutputTokens": 64,
                    "temperature": 0.0,
                }
                headers = {"Content-Type": "application/json"}
                try:
                    r = requests.post(url, json=body, headers=headers, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    # response structure for generativelanguage: 'candidates' -> list with 'output' or 'content' or 'content'
                    text = ""
                    if isinstance(data, dict):
                        if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"]):
                            cand = data["candidates"][0]
                            # try common fields
                            text = cand.get("output") or cand.get("content") or cand.get("displayText") or ""
                        else:
                            # some versions return 'output' directly
                            text = data.get("output") or data.get("content") or ""
                    text = (text or "").strip()
                    return text.split()[0] if text else ""
                except requests.RequestException as e:
                    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Gemini error: {str(e)}")

            # Provider selection
            try:
                if AI_PROVIDER == "openai":
                    word = call_openai(question)
                elif AI_PROVIDER == "anthropic":
                    word = call_anthropic(question)
                elif AI_PROVIDER == "gemini":
                    word = call_gemini(question)
                else:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown AI_PROVIDER")
                return JSONResponse(status_code=200, content=make_success(word))
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI service error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health():
    return JSONResponse(status_code=200, content={"is_success": True, "official_email": OFFICIAL_EMAIL})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

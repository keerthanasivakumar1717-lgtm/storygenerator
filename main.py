"""
StorySpark AI — Backend
FastAPI server that generates stories using Google Gemini API.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.8-flash"
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    print(
        "\n  WARNING: GEMINI_API_KEY is not set or still has the placeholder value.\n"
        "  Copy backend/.env.example to backend/.env and add your real key.\n"
        "  Get a key at https://aistudio.google.com/apikey\n"
    )

# Keep the app available when the key is missing; report the configuration error
# when a story is requested.
gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"
    else None
)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="StorySpark AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

VALID_GENRES = {"Fantasy", "Adventure", "Mystery", "Horror", "Comedy", "Romance"}
VALID_LENGTHS = {"Short", "Medium", "Long"}

LENGTH_GUIDANCE = {
    "Short": "about 300 words",
    "Medium": "about 700 words",
    "Long": "about 1500 words",
}


class StoryRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    genre: str = Field(..., min_length=1)
    length: str = Field(..., min_length=1)


class StoryResponse(BaseModel):
    story: str
    title: str
    genre: str
    length: str
    topic: str

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.post("/api/generate", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    """Generate a story using Google Gemini."""

    # --- Validate inputs ---
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    if request.genre not in VALID_GENRES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid genre. Choose from: {', '.join(sorted(VALID_GENRES))}",
        )

    if request.length not in VALID_LENGTHS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid length. Choose from: {', '.join(sorted(VALID_LENGTHS))}",
        )

    if gemini_client is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured. Add your key to backend/.env",
        )

    # --- Build prompt ---
    word_guide = LENGTH_GUIDANCE[request.length]
    prompt = (
        f"You are a talented fiction author. Write an original {request.genre.lower()} story "
        f"about the following topic: \"{topic}\".\n\n"
        f"Guidelines:\n"
        f"- Length: {word_guide}.\n"
        f"- Include a compelling title on the very first line, prefixed with 'Title: '.\n"
        f"- After the title, leave a blank line, then write the full story.\n"
        f"- Use vivid descriptions, engaging dialogue, and a satisfying ending.\n"
        f"- Do NOT include any meta-commentary — only output the title and story."
    )

    # --- Call Gemini ---
    try:
        interaction = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt,
        )
        raw_text = (interaction.output_text or "").strip()
        if not raw_text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response. Try a different topic.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to generate story: {str(exc)}",
        )

    # --- Parse title from response ---
    lines = raw_text.split("\n", 1)
    title = "Untitled Story"
    story_body = raw_text

    first_line = lines[0].strip()
    if first_line.lower().startswith("title:"):
        title = first_line[len("title:"):].strip().strip("*#").strip()
        story_body = lines[1].strip() if len(lines) > 1 else ""
    elif first_line.startswith("#"):
        title = first_line.lstrip("#").strip()
        story_body = lines[1].strip() if len(lines) > 1 else ""

    return StoryResponse(
        story=story_body,
        title=title,
        genre=request.genre,
        length=request.length,
        topic=topic,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

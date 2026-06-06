import re
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob


YOUTUBE_API_KEY = "AIzaSyCvpBYeMywHeKJVqh_Zh7a_xEjEiJKZYBk"
MAX_RESULTS_DEFAULT = 5

app = FastAPI(title="YouTube Comment Sentiment API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class YouTubeRequest(BaseModel):
    url: str = Field(...)
    max_comments: int = Field(default=MAX_RESULTS_DEFAULT, ge=1, le=50)

def extract_video_id(url: str) -> Optional[str]:
    url = url.strip()
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    return None

def sentiment_label(polarity: float) -> str:
    if polarity > 0.05:
        return "Positive"
    if polarity < -0.05:
        return "Negative"
    return "Neutral"

def analyze_text(text: str) -> Dict[str, Any]:
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment_label(polarity)
    }

def get_youtube_client():
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=400, detail="YouTube API key missing.")
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def fetch_recent_comments(video_id: str, max_comments: int) -> List[Dict[str, Any]]:
    youtube = get_youtube_client()
    try:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            order="time",
            textFormat="plainText"
        )
        resp = req.execute()
        items = resp.get("items", [])
        comments = []
        for it in items:
            snip = it["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": snip.get("authorDisplayName"),
                "text": snip.get("textDisplay"),
                "published_at": snip.get("publishedAt"),
                "like_count": snip.get("likeCount", 0)
            })
        return comments
    except HttpError as e:
        raise HTTPException(status_code=500, detail=f"YouTube API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch comments: {e}")

@app.get("/")
def root():
    return FileResponse("templates/index.html")

@app.post("/analyze_youtube/")
def analyze_youtube(req: YouTubeRequest):
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    comments = fetch_recent_comments(video_id, req.max_comments)
    if not comments:
        return {
            "video_id": video_id,
            "total_comments": 0,
            "results": []
        }

    results = []
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for c in comments:
        analysis = analyze_text(c["text"])
        counts[analysis["sentiment"]] += 1
        results.append({**c, **analysis})

    return {
        "video_id": video_id,
        "total_comments": len(results),
        "sentiment_counts": counts,
        "results": results
    }
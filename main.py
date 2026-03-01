import os
import io
import random
import re
import uuid
import secrets
from time import time
from datetime import datetime, timedelta
from collections import Counter

from fastapi import FastAPI, UploadFile, File, HTTPException, Cookie, Depends, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pypdf import PdfReader
from supabase import create_client, Client
import google.generativeai as genai
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.sessions import SessionMiddleware

# -----------------------
# Load ENV
# -----------------------
load_dotenv()

CACHE_TTL = 300  # เก็บ 5 นาที
response_cache = {}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")

ADMIN_USER = os.getenv("ADMIN_USERNAME")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Missing Supabase config")

if not API_KEYS or API_KEYS == [""]:
    raise Exception("No GEMINI_API_KEYS found")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# เริ่มสุ่ม key
current_index = random.randint(0, len(API_KEYS) - 1)

def use_key(index):
    genai.configure(api_key=API_KEYS[index].strip())

# -----------------------
# FastAPI Setup
# -----------------------
app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key="SUPER_SECRET_KEY_CHANGE_THIS"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "คุณส่งข้อความถี่เกินไป กรุณารอสักครู่"}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic(auto_error=False)

def verify_admin(request: Request):
    if request.session.get("admin"):
        return True

    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        raise HTTPException(status_code=302, headers={"Location": "/login"})

    raise HTTPException(status_code=401, detail="Unauthorized")
# -----------------------
# Utility
# -----------------------

def split_text(text, chunk_size=1000, overlap=200):
    # ทำความสะอาดข้อความเบื้องต้น
    text = text.replace("\r", "")
    text = re.sub(r'\n{2,}', '\n\n', text)

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # ถ้าย่อหน้ายาวเกิน chunk_size → ตัดย่อย
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            for i in range(0, len(para), chunk_size):
                sub = para[i:i+chunk_size]
                chunks.append(sub.strip())
            continue

        # ถ้าเพิ่มแล้วเกิน chunk_size → ปิด chunk เดิม
        if len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    # ------------------------
    # เพิ่ม overlap
    # ------------------------
    final_chunks = []

    for i in range(len(chunks)):
        if i == 0:
            final_chunks.append(chunks[i])
        else:
            prev = final_chunks[-1]
            overlap_text = prev[-overlap:]
            combined = overlap_text + "\n" + chunks[i]
            final_chunks.append(combined.strip())

    print("TOTAL CHUNKS:", len(final_chunks))
    return final_chunks


def embed_text(text):
    global current_index
    last_error = None

    for i in range(len(API_KEYS)):
        idx = (current_index + i) % len(API_KEYS)
        try:
            use_key(idx)
            result = genai.embed_content(
                model="gemini-embedding-001",
                content=text
            )
            current_index = idx
            return result["embedding"]
        except Exception as e:
            print(f"Embed failed with key {idx}: {e}")
            last_error = e

    raise last_error


GENERATION_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash"
]

def generate_answer(prompt):
    global current_index
    last_error = None

    # วนทุก API key
    for k in range(len(API_KEYS)):
        key_index = (current_index + k) % len(API_KEYS)

        try:
            use_key(key_index)

            # วนทุกโมเดล
            for model_name in GENERATION_MODELS:
                try:
                    print(f"Trying model: {model_name} with key {key_index}")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)

                    current_index = key_index
                    return response.text

                except Exception as model_error:
                    print(f"Model {model_name} failed:", model_error)
                    last_error = model_error

        except Exception as key_error:
            print(f"Key {key_index} failed:", key_error)
            last_error = key_error

    raise last_error

def process_pdf_background(file_path, filename, category):
    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text: full_text += text + "\n"
        
        chunks = split_text(full_text)
        
        for chunk in chunks:
            embedding = embed_text(chunk)
            supabase.table("documents").insert({
                "content": chunk,
                "embedding": embedding,
                "source": filename,
                "category": category
            }).execute()
        print(f"Processed {filename} successfully.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# -----------------------
# Root Check
# -----------------------
@app.get("/")
def home():
    return FileResponse("static/chat.html")

@app.get("/login")
def login_page():
    return FileResponse("static/login.html")

@app.get("/admin", dependencies=[Depends(verify_admin)])
def admin_page():
    return FileResponse("static/admin.html")

@app.get("/admin/stats", dependencies=[Depends(verify_admin)])
def stats_page():
    return FileResponse("static/stats.html")

@app.post("/api/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == ADMIN_USER and password == ADMIN_PASS:
        request.session["admin"] = username
        return {"success": True}

    raise HTTPException(status_code=401, detail="Invalid credentials")
# -----------------------
# Upload PDF (Admin)
# -----------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-pdf", dependencies=[Depends(verify_admin)])
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(...)
):
    try:
        contents = await file.read()

        # บันทึกไฟล์ลง disk
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # 🟢 โยนงานประมวลผลไปทำเป็น Background Task
        background_tasks.add_task(process_pdf_background, file_path, file.filename, category)

        # ✅ ตอบกลับ Frontend ทันที ไม่ต้องรอให้แปลงไฟล์เสร็จ
        return {"message": "กำลังประมวลผลไฟล์ในพื้นหลัง"}    

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/pdfs")
def list_pdfs():
    result = supabase.table("documents") \
        .select("source, category") \
        .execute()

    files = {}
    for row in result.data:
        if row["source"] not in files:
            files[row["source"]] = row["category"]

    return {"files": files}

@app.get("/pdfs/{filename}")
def get_pdf(filename: str):
    file_path = os.path.join("uploads", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์")

    return FileResponse(
        file_path,
        media_type="application/pdf"
    )

@app.get("/pdfs/download/{filename}")
def download_pdf(filename: str):
    file_path = os.path.join("uploads", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์")

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )

@app.delete("/pdfs/{filename}")
def delete_pdf(filename: str):

    # ลบจาก DB
    supabase.table("documents") \
        .delete() \
        .eq("source", filename) \
        .execute()

    # ลบไฟล์จริง
    file_path = os.path.join("uploads", filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"message": "ลบสำเร็จ"}


# -----------------------
# Chat Endpoint
# -----------------------
@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, session_id: str = Cookie(default=None)):
    try:
        body = await request.json()
        question = body.get("message", "").strip()

        if not question:
            return {"answer": "กรุณาพิมพ์คำถามก่อนส่งค่ะ"}

        if len(question) > 500:
            return {"answer": "ข้อความยาวเกินไป กรุณาส่งไม่เกิน 500 ตัวอักษรค่ะ"}

        # Session management
        if session_id:
            check = supabase.table("chat_sessions").select("id").eq("id", session_id).execute()
            if not check.data:
                session_id = None

        if not session_id:
            session = supabase.table("chat_sessions").insert({}).execute()
            session_id = session.data[0]["id"]

        # Cache (per session)
        cache_key = f"{session_id}:{question.lower()}"
        now_ts = time()

        # Evict expired cache entries
        expired = [k for k, v in response_cache.items() if now_ts - v["timestamp"] > CACHE_TTL]
        for k in expired:
            del response_cache[k]

        if cache_key in response_cache:
            resp = JSONResponse({"answer": response_cache[cache_key]["answer"]})
            resp.set_cookie("session_id", session_id, httponly=True, max_age=60 * 60 * 24 * 30)
            return resp

        # Save user message
        supabase.table("chat_messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": question
        }).execute()

        # Get history
        history_result = supabase.table("chat_messages") \
            .select("role,content") \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()

        history = history_result.data or []

        # Summarize if too long
        summary = ""
        if len(history) > 12:
            conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
            summary_prompt = f"สรุปบทสนทนานี้ให้สั้น กระชับ และเก็บประเด็นสำคัญ:\n\n{conversation_text}"
            summary = generate_answer(summary_prompt)
            supabase.table("chat_summaries").upsert({
                "session_id": session_id,
                "summary": summary
            }).execute()

        if len(history) > 10:
            history = history[-10:]

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

        # Rewrite question with context
        rewritten_question = question
        if history_text:
            rewrite_prompt = f"""จากบทสนทนานี้:
{history_text}

เขียนคำถามล่าสุดใหม่ให้ชัดเจน:
{question}"""
            rewritten_question = generate_answer(rewrite_prompt)

        print(f"REWRITTEN: {rewritten_question}")

        # RAG
        question_embedding = embed_text(rewritten_question)
        result = supabase.rpc("match_documents", {
            "query_embedding": question_embedding,
            "match_threshold": 0.5,
            "match_count": 5
        }).execute()

        matches = result.data
        categories = list(set([m["category"] for m in matches if m.get("category")]))
        main_category = categories[0] if categories else "อื่น ๆ"

        context = "\n".join([m["content"] for m in matches]) if matches else ""
        if summary:
            context = f"สรุปบทสนทนาก่อนหน้า:\n{summary}\n\n" + context

        extra_context = f"บทสนทนาก่อนหน้า:\n{history_text}\n\n" if history_text else ""

        prompt = f"""คุณคือแชทบอทเทศบาล เป็นบอทที่คอยช่วยตอบคำถามให้กับประชาชนที่เข้ามาสอบถาม
กติกาสำคัญ:
- ให้ใช้ข้อมูลจาก "ข้อมูลเอกสาร" เป็นหลักในการตอบ
- สามารถใช้ "บทสนทนาก่อนหน้า" เพื่อทำความเข้าใจคำถามอ้างอิง
- ห้ามแต่งข้อมูลที่ไม่มีในข้อมูลเอกสาร
- ถ้าไม่มีข้อมูลจริง ๆ ให้ตอบว่าไม่พบข้อมูล
- ตอบเป็น Markdown ได้ (ใช้ **ตัวหนา**, รายการ - ได้)

ข้อมูลเอกสาร:
{context}

{extra_context}
คำแนะนำ:
1. ถ้าเป็นคำทักทายหรือกล่าวลา ตอบอย่างสุภาพและเป็นมิตร
2. ตอบให้กระชับและเป็นกันเอง
3. ไม่ต้องสวัสดีทุกรอบ
4. แทน User ว่า "คุณ" เสมอ
5. ห้ามตอบเรื่องศาสนา การเมือง พระมหากษัตริย์

คำถาม: {question}"""

        answer = generate_answer(prompt)

        # Cache answer
        response_cache[cache_key] = {"answer": answer, "timestamp": time()}

        # Analytics (เก็บเฉพาะเมื่อหมวดเปลี่ยน)
        last_cat_result = supabase.table("chat_analytics") \
            .select("category") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        last_cat = last_cat_result.data[0]["category"] if last_cat_result.data else None

        if not last_cat or last_cat != main_category:
            supabase.table("chat_analytics").insert({
                "session_id": session_id,
                "question": question,
                "category": main_category
            }).execute()

        # Save bot message
        supabase.table("chat_messages").insert({
            "session_id": session_id,
            "role": "assistant",
            "content": answer
        }).execute()

        resp = JSONResponse({"answer": answer})
        resp.set_cookie("session_id", session_id, httponly=True, max_age=60 * 60 * 24 * 30)
        return resp

    except Exception as e:
        print(f"CHAT ERROR: {e}")
        return {"error": "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ"}


@app.get("/analytics/categories-list")
def categories_list():
    result = supabase.table("chat_analytics") \
        .select("category") \
        .execute()

    categories = list(set(
        [row["category"] for row in result.data if row.get("category")]
    ))

    return categories

@app.get("/analytics/summary")
def analytics_summary(start: str = None, end: str = None, category: str = "all"):

    query = supabase.table("chat_analytics").select("*")

    if start:
        query = query.gte("created_at", start)
    if end:
        query = query.lte("created_at", end)
    if category != "all":
        query = query.eq("category", category)

    result = query.execute()

    total_questions = len(result.data or [])
    
    # นับ session ไม่ซ้ำ
    sessions = set(
        [row["session_id"] for row in result.data or []]
    )

    categories = set(
        [row["category"] for row in result.data or []]
    )

    return {
        "total_questions": total_questions,
        "total_users": len(sessions),
        "total_categories": len(categories)
    }

@app.get("/analytics/last7days")
def analytics_last7days(category: str = "all"):

    from datetime import datetime, timedelta
    from collections import Counter

    result = supabase.table("chat_analytics") \
        .select("created_at, category") \
        .execute()

    counter = Counter()
    today = datetime.utcnow().date()

    for row in result.data or []:
        date = row["created_at"][:10]
        cat = row["category"]

        if category != "all" and cat != category:
            continue

        counter[date] += 1

    data = []

    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        date_str = str(day)
        data.append({
            "date": date_str,
            "count": counter.get(date_str, 0)
        })

    return data

@app.get("/analytics/top-questions")
def analytics_top_questions(category: str = "all"):

    from collections import Counter

    query = supabase.table("chat_analytics") \
        .select("question, category") \
        .execute()

    counter = Counter()

    for row in query.data or []:
        if category != "all" and row["category"] != category:
            continue

        question = row["question"]
        counter[question] += 1

    top = counter.most_common(10)

    return [
        {"question": q, "count": c}
        for q, c in top
    ]

def get_recent_messages(session_id, limit=10):
    result = supabase.table("chat_messages") \
        .select("role,content") \
        .eq("session_id", session_id) \
        .order("created_at", desc=False) \
        .limit(limit) \
        .execute()

    return result.data

def save_message(session_id, role, content):
    supabase.table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()

def get_summary(session_id):
    result = supabase.table("chat_summaries") \
        .select("*") \
        .eq("session_id", session_id) \
        .execute()

    return result.data[0]["summary"] if result.data else ""

def update_summary(session_id, messages):
    conversation_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages]
    )

    prompt = f"""
สรุปบทสนทนานี้ให้สั้น กระชับ และเก็บประเด็นสำคัญ:

{conversation_text}
"""

    summary = generate_answer(prompt)

    supabase.table("chat_summaries").upsert({
        "session_id": session_id,
        "summary": summary
    }).execute()

    return summary

@app.get("/analytics/users")
def daily_users():
    result = supabase.table("chat_sessions") \
        .select("created_at") \
        .execute()

    from collections import Counter
    from datetime import datetime

    counter = Counter()

    for row in result.data:
        date = row["created_at"][:10]
        counter[date] += 1

    data = [{"date": k, "users": v} for k, v in sorted(counter.items())]
    return data

@app.get("/analytics/categories")
def category_stats():
    try:
        result = supabase.table("chat_analytics") \
            .select("category") \
            .execute()

        from collections import Counter
        counter = Counter()

        for row in result.data or []:
            category = row["category"] or "ไม่ระบุ"
            counter[category] += 1

        data = [{"category": k, "count": v} for k, v in counter.items()]
        return data

    except Exception as e:
        print("ANALYTICS ERROR:", e)
        return []
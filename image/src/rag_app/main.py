import warnings
import logging
import os
import shutil
import zipfile
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import List
import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sqlalchemy import func
from dotenv import load_dotenv

# Import our modularized files
import models
from database import engine, SessionLocal, get_db
from schemas import ChatRequest, LeadSubmitRequest, LeadUpdateRequest, BulkLeadContactedRequest
from auth import (
    get_current_admin, get_dashboard_user, get_trackdashboard_user,
    _hash_password, _make_session_token
)
from core_ai import (
    reload_vector_store, get_chroma_stats, CHROMA_PATH,
    generate_response_stream, _compute_lead_status, ingest_pdfs
)
# ---------------------------------------------------------------------------
# Suppress noisy logs
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*Failed to send telemetry event.*")
warnings.filterwarnings("ignore", message=".*telemetry.*")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MAINTENANCE_MODE = False

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="RAG API Final System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ---------------------------------------------------------------------------
# Lead scoring helper
# ---------------------------------------------------------------------------
_MONTHS_AR = ["يناير","فبراير","مارس","أبريل","مايو","يونيو","يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]
_DAYS_AR = ["الاثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت","الأحد"]
_WEEK_ORDINALS_AR = ["الأول","الثاني","الثالث","الرابع","الخامس"]

def _week_label(year: int, week_num: int) -> str:
    from datetime import datetime
    start = datetime.fromisocalendar(year, week_num, 1)
    week_of_month = (start.day - 1) // 7
    return f"الأسبوع {_WEEK_ORDINALS_AR[week_of_month]} من {_MONTHS_AR[start.month - 1]}"


def _week_bounds(year: int, week_num: int) -> tuple:
    """Start/end datetimes (inclusive) for an ISO week — for SQL range filtering."""
    from datetime import datetime, timedelta
    start = datetime.fromisocalendar(year, week_num, 1).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start, end


def _month_bounds(year: int, month: int) -> tuple:
    """Start/end datetimes (inclusive) for a calendar month — for SQL range filtering."""
    from datetime import datetime
    from calendar import monthrange
    start = datetime(year, month, 1)
    last_day = monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59)
    return start, end


def _build_period_label(report_type: str, filename: str, req) -> tuple:
    """Derive (period_label, report_period) from the uploaded report's filename."""
    from datetime import date
    today = date.today()
    fname = filename.lower()
    if "monthly" in fname or "month" in fname:
        return f"شهري — {_MONTHS_AR[today.month - 1]} {req.year}", "monthly"
    if report_type == "repeated_visitors" and ("daily" in fname or "day" in fname):
        return f"يومي — {today.strftime('%Y-%m-%d')}", "daily"
    return f"أسبوعي — أسبوع {req.week_number} / {req.year}", "weekly"




#===================================================
# to get all unanswered questions for the current week
#ُ=====================================================
@app.get("/api/unanswered-questions")
async def get_unanswered_questions(
    request: Request,
    week: int = None,
    year: int = None,
    db: Session = Depends(get_db),
):
    get_dashboard_user(request)

    from datetime import date
    from collections import Counter

    if not week: week = date.today().isocalendar()[1]
    if not year: year = date.today().year

    start_of_week, end_of_week = _week_bounds(year, week)
    week_unanswered = db.query(models.ChatLog).filter(
        models.ChatLog.is_unanswered == True,
        models.ChatLog.timestamp >= start_of_week,
        models.ChatLog.timestamp <= end_of_week,
    ).all()

    counter = Counter(log.user_query for log in week_unanswered)

    return [
        {"query": q, "count": c}
        for q, c in counter.most_common(20)
    ]


# ===========================================================================
# EXISTING ROUTES (unchanged)
# ===========================================================================

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    if MAINTENANCE_MODE:
        return templates.TemplateResponse("user_maintenance.html", {"request": request})
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/chat-en", response_class=HTMLResponse)
async def chat_page_en(request: Request):
    if MAINTENANCE_MODE:
        return templates.TemplateResponse("user_maintenance_en.html", {"request": request})
    return templates.TemplateResponse("chat_en.html", {"request": request})


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    if MAINTENANCE_MODE:
        async def msg():
            yield "Sorry, the system is under maintenance. Please refresh the page."
        return StreamingResponse(msg(), media_type="text/plain")

    return StreamingResponse(
        generate_response_stream(request.message, request.history, request.session_id, db),
        media_type="text/plain",
    )


@app.get("/admin/maintenance", response_class=HTMLResponse)
def maintenance_page(request: Request, username: str = Depends(get_current_admin)):
    stats = get_chroma_stats()
    return templates.TemplateResponse(
        "maintenance.html",
        {"request": request, "is_maintenance": MAINTENANCE_MODE, "db_stats": stats},
    )


@app.post("/admin/toggle-maintenance")
async def toggle_maintenance(request: Request, username: str = Depends(get_current_admin)):
    global MAINTENANCE_MODE
    form_data = await request.form()
    state = form_data.get("state")
    if state == "on":
        MAINTENANCE_MODE = True
    elif state == "off":
        MAINTENANCE_MODE = False

    stats = get_chroma_stats()
    return templates.TemplateResponse(
        "maintenance.html",
        {"request": request, "is_maintenance": MAINTENANCE_MODE, "db_stats": stats},
    )


def _replace_chroma_db(file_obj) -> dict:
    """Blocking disk I/O for a KB upload — run via run_in_threadpool."""
    temp_zip = "temp.zip"
    with open(temp_zip, "wb") as b:
        shutil.copyfileobj(file_obj, b)

    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except Exception:
            pass

    CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(temp_zip, "r") as z:
        z.extractall(CHROMA_PATH.parent)

    os.remove(temp_zip)
    reload_vector_store()
    return get_chroma_stats()


@app.post("/admin/upload-db")
async def upload_db(
    request: Request,
    file: UploadFile = File(...),
    username: str = Depends(get_current_admin),
):
    global MAINTENANCE_MODE
    MAINTENANCE_MODE = True
    try:
        stats = await run_in_threadpool(_replace_chroma_db, file.file)
        return templates.TemplateResponse(
            "maintenance.html",
            {
                "request": request,
                "is_maintenance": False,
                "db_stats": stats,
                "message": "Knowledge base updated successfully!",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        MAINTENANCE_MODE = False


@app.post("/admin/upload-pdfs")
async def upload_pdfs(
    request: Request,
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    username: str = Depends(get_current_admin),
):
    if not files or all(f.filename == "" for f in files):
        return JSONResponse(status_code=400, content={"error": "No PDF files provided"})

    pdf_bytes_list = []
    filenames = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": f"{f.filename} is not a PDF"})
        pdf_bytes_list.append(await f.read())
        filenames.append(f.filename)

    try:
        stats = await run_in_threadpool(ingest_pdfs, pdf_bytes_list, filenames, chunk_size, chunk_overlap)
        db_stats = get_chroma_stats()
        return templates.TemplateResponse(
            "maintenance.html",
            {
                "request": request,
                "is_maintenance": MAINTENANCE_MODE,
                "db_stats": db_stats,
                "message": f"✓ Ingested {stats['files']} file(s) — {stats['pages']} pages — {stats['chunks']} chunks added.",
            },
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/admin/kpi", response_class=HTMLResponse)
def kpi_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_admin),
):
    logs = (
        db.query(models.ChatLog)
        .order_by(models.ChatLog.timestamp.desc())
        .limit(50)
        .all()
    )

    grouped_sessions = {}
    for log in logs:
        sid = log.session_id or "Anonymous"
        if sid not in grouped_sessions:
            grouped_sessions[sid] = []
        grouped_sessions[sid].append(log)

    total_chats = db.query(models.ChatLog).count()

    avg_speed_all = 0
    avg_speed_last_10 = 0

    if total_chats > 0:
        all_logs = db.query(models.ChatLog.response_time).all()
        all_times = [r[0] for r in all_logs if r[0] is not None]
        if all_times:
            avg_speed_all = sum(all_times) / len(all_times)

        last_10_logs = (
            db.query(models.ChatLog.response_time)
            .order_by(models.ChatLog.timestamp.desc())
            .limit(10)
            .all()
        )
        last_10_times = [r[0] for r in last_10_logs if r[0] is not None]
        if last_10_times:
            avg_speed_last_10 = sum(last_10_times) / len(last_10_times)

    return templates.TemplateResponse(
        "kpi.html",
        {
            "request": request,
            "grouped_sessions": grouped_sessions,
            "logs": logs,
            "total_chats": total_chats,
            "avg_speed_all": round(avg_speed_all, 2),
            "avg_speed_last_10": round(avg_speed_last_10, 2),
            "now": datetime.now(),
        },
    )


@app.get("/admin/kpi/{log_id}", response_class=HTMLResponse)
def view_chat_log(
    log_id: int,
    request: Request,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_admin),
):
    log = db.query(models.ChatLog).filter(models.ChatLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    return templates.TemplateResponse("chat_details.html", {"request": request, "log": log})


def _fetch_cloudwatch_rag_eval(limit: int = 30):
    """Fetch the most recent RAG_EVAL log entries via CloudWatch Logs Insights.

    Returns (events, error) — error is None on success, or a short message
    if CloudWatch is unreachable (e.g. no AWS credentials in local dev).

    Uses Logs Insights (same query documented in CLAUDE.md) instead of
    tailing raw get_log_events: RAG_EVAL lines are sparse compared to the
    routine uvicorn access-log noise (health checks, /docs polling) in the
    stream, so a plain "last N lines" tail can easily miss them entirely.
    Insights filters server-side across the whole log group instead.
    """
    import time as _time

    try:
        import boto3

        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        client = boto3.client("logs", region_name=region)
        log_group = os.getenv("CLOUDWATCH_LOG_GROUP", "/ecs/sstli-chatbot")

        end_time = int(_time.time())
        start_time = end_time - 7 * 24 * 3600  # look back 7 days

        query_id = client.start_query(
            logGroupName=log_group,
            startTime=start_time,
            endTime=end_time,
            queryString=(
                'fields @timestamp, question, answer '
                '| filter log_type = "RAG_EVAL" '
                '| sort @timestamp desc '
                f'| limit {limit}'
            ),
        )["queryId"]

        result = None
        for _ in range(15):  # poll up to ~7.5s
            result = client.get_query_results(queryId=query_id)
            if result["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
                break
            _time.sleep(0.5)

        if not result or result["status"] != "Complete":
            status = result["status"] if result else "no response"
            return [], f"CloudWatch query did not complete ({status})."

        events = []
        for row in result.get("results", []):
            field_map = {f["field"]: f["value"] for f in row}
            events.append({
                "timestamp": field_map.get("@timestamp"),
                "question": field_map.get("question", ""),
                "answer": field_map.get("answer", ""),
            })
        return events, None
    except Exception as e:
        return [], f"CloudWatch unavailable: {e}"


@app.get("/admin/monitoring", response_class=HTMLResponse)
def monitoring_page(
    request: Request,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_admin),
):
    logs = (
        db.query(models.ChatLog)
        .order_by(models.ChatLog.timestamp.desc())
        .limit(100)
        .all()
    )

    total_chats = db.query(models.ChatLog).count()
    unanswered_count = db.query(models.ChatLog).filter(models.ChatLog.is_unanswered == True).count()

    avg_speed_all = 0
    avg_speed_last_10 = 0
    if total_chats > 0:
        all_times = [r[0] for r in db.query(models.ChatLog.response_time).all() if r[0] is not None]
        if all_times:
            avg_speed_all = sum(all_times) / len(all_times)

        last_10_times = [
            r[0] for r in db.query(models.ChatLog.response_time)
            .order_by(models.ChatLog.timestamp.desc())
            .limit(10)
            .all()
            if r[0] is not None
        ]
        if last_10_times:
            avg_speed_last_10 = sum(last_10_times) / len(last_10_times)

    cloudwatch_events, cloudwatch_error = _fetch_cloudwatch_rag_eval()

    return templates.TemplateResponse(
        "monitoring.html",
        {
            "request": request,
            "logs": logs,
            "total_chats": total_chats,
            "unanswered_count": unanswered_count,
            "avg_speed_all": round(avg_speed_all, 2),
            "avg_speed_last_10": round(avg_speed_last_10, 2),
            "cloudwatch_events": cloudwatch_events,
            "cloudwatch_error": cloudwatch_error,
        },
    )


@app.get("/admin/export-csv")
def export_logs_csv(
    db: Session = Depends(get_db), username: str = Depends(get_current_admin)
):
    logs = db.query(models.ChatLog).order_by(models.ChatLog.timestamp.desc()).all()
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow(["ID", "Date", "Time", "User Question", "Bot Answer", "Response Time (s)"])
    for log in logs:
        writer.writerow([
            log.id,
            log.timestamp.strftime("%Y-%m-%d"),
            log.timestamp.strftime("%H:%M:%S"),
            log.user_query,
            log.bot_answer,
            f"{log.response_time:.2f}",
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chat_logs_report.csv"},
    )


@app.get("/admin/full-report")
def full_report_page(
    request: Request,
    download: bool = False,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_admin),
):
    logs = db.query(models.ChatLog).order_by(models.ChatLog.timestamp.desc()).all()
    context = {"request": request, "logs": logs, "generated_at": datetime.now()}
    if download:
        html_content = templates.get_template("full_report.html").render(context)
        return HTMLResponse(
            content=html_content,
            headers={"Content-Disposition": f"attachment; filename=full_report_{datetime.now().strftime('%Y-%m-%d')}.html"},
        )
    return templates.TemplateResponse("full_report.html", context)


@app.delete("/admin/kpi/delete/{log_id}")
def delete_chat_log(
    log_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_admin),
):
    
    log = db.query(models.ChatLog).filter(models.ChatLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    db.delete(log)
    db.commit()
    return {"status": "success", "message": f"Log {log_id} deleted"}


@app.get("/admin/db-info")
def db_info_endpoint(username: str = Depends(get_current_admin)):
    return {"maintenance_mode": MAINTENANCE_MODE, "db_stats": get_chroma_stats()}


@app.delete("/admin/kpi/delete-session/{session_id}")
def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_admin),
):
    rows_deleted = (
        db.query(models.ChatLog)
        .filter(models.ChatLog.session_id == session_id)
        .delete()
    )
    db.commit()
    if rows_deleted == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": f"Session {session_id} deleted ({rows_deleted} logs)"}


# ===========================================================================
# NEW ROUTE: Lead submission from chat popup
# ===========================================================================

@app.post("/api/lead/submit")
async def submit_lead(payload: LeadSubmitRequest, db: Session = Depends(get_db)):
    """
    Called from chat.js when the user submits their phone number.
    Creates or updates a Lead record for the session.
    Starts as pending (is_approved=False) until admin approves.
    """
    existing = db.query(models.Lead).filter(models.Lead.session_id == payload.session_id).first()
    
    if existing:
        existing.phone_number = payload.phone_number
        existing.question_count = payload.question_count
        
        # Update the new fields if provided in the payload
        if payload.is_registered is not None:
            existing.is_registered = payload.is_registered
        if payload.city is not None:
            existing.city = payload.city
            
        # Don't override asked_about_price/reg - managed by LLM in background
        existing.lead_status = _compute_lead_status(
            payload.question_count, existing.asked_about_price, existing.asked_about_registration
        )
        db.commit()
        return {"status": "updated", "lead_id": existing.id}

    # Create new lead if it doesn't exist
    lead = models.Lead(
        session_id=payload.session_id,
        phone_number=payload.phone_number,
        is_registered=payload.is_registered,  # New field
        city=payload.city,                    # New field
        question_count=payload.question_count,
        asked_about_price=False,
        asked_about_registration=False,
        lead_status=_compute_lead_status(
            payload.question_count, False, False
        ),
        is_approved=False,
    )
    
    db.add(lead)
    db.commit()
    db.refresh(lead)
    
    return {"status": "created", "lead_id": lead.id}


# ===========================================================================
# NEW ROUTES: Dashboard login / logout
# ===========================================================================

@app.get("/dashboard/login", response_class=HTMLResponse)
async def dashboard_login_page(request: Request):
    return templates.TemplateResponse("dashboard_login.html", {"request": request, "error": None})


@app.post("/dashboard/login")
async def dashboard_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    hashed = _hash_password(password)
    user = (
        db.query(models.DashboardUser)
        .filter(
            models.DashboardUser.username == username,
            models.DashboardUser.hashed_password == hashed,
            models.DashboardUser.is_active == True,
        )
        .first()
    )
    if not user:
        return templates.TemplateResponse(
            "dashboard_login.html",
            {"request": request, "error": "Invalid username or password."},
        )

    # Update last login timestamp
    user.last_login = datetime.now()
    db.commit()

    token = _make_session_token(user.username, user.role)
    redirect_url = "/trackdashboard" if user.role == "admin" else "/dashboard"
    response = RedirectResponse(url=redirect_url, status_code=302)
    response.set_cookie(
        key="dashboard_session",
        value=token,
        httponly=True,
        max_age=60 * 60 * 8,  # 8 hours
        samesite="lax",
    )
    return response


@app.get("/dashboard/logout")
async def dashboard_logout():
    response = RedirectResponse(url="/dashboard/login", status_code=302)
    response.delete_cookie("dashboard_session")
    return response


# ===========================================================================
# NEW ROUTES: /dashboard  (sales team view — approved data only)
# ===========================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def sales_dashboard(request: Request, db: Session = Depends(get_db)):
    try:
        username, role = get_dashboard_user(request)
    except HTTPException:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    from datetime import datetime, timedelta
    from collections import Counter

    # Get current date to load the current week on first page visit
    now = datetime.utcnow()
    year = now.year
    week_num = now.isocalendar()[1]

    # Calculate start and end of the current week
    start_of_week = now - timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)

    # 1. Fetch leads for top stats (current week only)
    week_leads = db.query(models.Lead).filter(
        models.Lead.timestamp >= start_of_week,
        models.Lead.timestamp <= end_of_week
    ).all()

    week_session_count = len(week_leads)
    week_hot = sum(1 for l in week_leads if l.lead_status == "hot")
    week_warm = sum(1 for l in week_leads if l.lead_status == "warm")
    week_maxq = max((l.question_count for l in week_leads), default=0)

    # 2. Fetch all leads for the table (current week only)
    leads = (
        db.query(models.Lead)
        .filter(
            models.Lead.timestamp >= start_of_week,
            models.Lead.timestamp <= end_of_week
        )
        .order_by(models.Lead.timestamp.desc())
        .all()
    )

    # 3. Fetch active weekly note
    weekly_note = (
        db.query(models.WeeklyNote)
        .filter(
            models.WeeklyNote.week_number == week_num,
            models.WeeklyNote.year == year,
            models.WeeklyNote.is_published == True,
        )
        .first()
    )

    # 4. Fetch peak reports
    peak_reports = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "peak_hours",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year        == year,
        )
        .all()
    )

    # 5. Fetch visible sections settings
    sections = {
        s.section_key: s.is_visible
        for s in db.query(models.DashboardSection).all()
    }

    # 6. Check if a questions report exists for the current week
    questions_report = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "questions",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year == year,
        )
        .first()
    )

    # 7. Top questions from approved sessions only (current week)
    session_ids = [l.session_id for l in leads]
    top_questions = []
    
    if session_ids:
        week_queries = (
            db.query(models.ChatLog.user_query)
            .filter(
                models.ChatLog.session_id.in_(session_ids),
                models.ChatLog.timestamp >= start_of_week,
                models.ChatLog.timestamp <= end_of_week
            )
            .all()
        )
        
        counter = Counter(q[0] for q in week_queries if q[0])     
        # Format as expected by Jinja template on initial load
        top_questions = [
            type('obj', (object,), {'user_query': q, 'cnt': c})()
            for q, c in counter.most_common()
        ]
    # Leads report for current week
    leads_report = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "leads",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year        == year,
        )
        .order_by(models.UploadedReport.uploaded_at.desc())
        .first()
    )
    # Repeated visitors reports (all for this week)
    repeated_reports_raw = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "repeated_visitors",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year        == year,
        )
        .order_by(models.UploadedReport.uploaded_at.desc())
        .all()
    )
    # Render template with initial data
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request":             request,
            "username":            username,
            "leads":               leads,
            "weekly_note":         weekly_note,
            "sections":            sections,
            "top_questions":       top_questions,
            "week_num":            week_num,
            "year_num":            year,
            "week_label":          _week_label(year, week_num),
            "week_session_count":  week_session_count,
            "has_questions_report": questions_report is not None,
            "week_hot":  week_hot,
            "week_warm": week_warm,
            "week_maxq": week_maxq,

            "peak_reports": [
                {
                    "id":           r.id,
                    "period_label": r.period_label,
                    "uploaded_at":  r.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                }
                for r in peak_reports
            ],
            "leads_report": leads_report,
            "repeated_reports": [
                {
                    "id":           r.id,
                    "period_label": r.period_label or f"أسبوع {week_num} / {year}",
                    "uploaded_at":  r.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                }
                for r in repeated_reports_raw
            ],
        }
    )
# ===========================================================================
# NEW ROUTES: /trackdashboard  (admin control panel)
# ===========================================================================

@app.get("/trackdashboard", response_class=HTMLResponse)
async def track_dashboard(request: Request, db: Session = Depends(get_db)):
    try:
        username, role = get_trackdashboard_user(request)
    except HTTPException:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    # All leads (pending + approved)
    leads = db.query(models.Lead).order_by(models.Lead.timestamp.desc()).all()

    # Dashboard users
    users = db.query(models.DashboardUser).order_by(models.DashboardUser.created_at.desc()).all()

    # Sections
    sections = db.query(models.DashboardSection).all()

    # Weekly notes
    from datetime import date
    week_num = date.today().isocalendar()[1]
    year = date.today().year
    weekly_note = (
        db.query(models.WeeklyNote)
        .filter(
            models.WeeklyNote.week_number == week_num,
            models.WeeklyNote.year == year,
        )
        .first()
    )

    # Stats summary
    total_leads = db.query(models.Lead).count()
    hot_leads = db.query(models.Lead).filter(models.Lead.lead_status == "hot").count()

    return templates.TemplateResponse(
        "trackdashboard.html",
        {
            "request": request,
            "username": username,
            "leads": leads,
            "users": users,
            "sections": sections,
            "weekly_note": weekly_note,
            "week_num": week_num,
            "total_leads": total_leads,
            "hot_leads": hot_leads,
            "now": datetime.now(),
        },
    )


# ---------------------------------------------------------------------------
# trackdashboard: reject a lead
# ---------------------------------------------------------------------------

@app.post("/trackdashboard/leads/{lead_id}/reject")
async def reject_lead(lead_id: int, request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    lead = db.query(models.Lead).filter(models.Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    db.delete(lead)
    db.commit()
    return {"status": "rejected", "lead_id": lead_id}


@app.patch("/trackdashboard/leads/{lead_id}/note")
async def update_lead_note(
    lead_id: int, payload: LeadUpdateRequest, request: Request, db: Session = Depends(get_db)
):
    get_dashboard_user(request) 
    lead = db.query(models.Lead).filter(models.Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    if payload.admin_note is not None:
        lead.admin_note = payload.admin_note
    if payload.lead_status is not None:
        lead.lead_status = payload.lead_status
    if payload.is_contacted is not None:
        lead.is_contacted = payload.is_contacted

    db.commit()
    return {"status": "updated"}


@app.patch("/trackdashboard/leads/bulk-contacted")
async def bulk_update_leads_contacted(
    payload: BulkLeadContactedRequest, request: Request, db: Session = Depends(get_db)
):
    """Update is_contacted for many leads in a single DB commit.

    Replaces firing one PATCH-per-lead from the frontend, which serializes
    behind SQLite's single-writer lock and gets slow fast as the selection grows.
    """
    get_dashboard_user(request)
    if not payload.lead_ids:
        return {"status": "updated", "updated_count": 0}

    updated_count = (
        db.query(models.Lead)
        .filter(models.Lead.id.in_(payload.lead_ids))
        .update({"is_contacted": payload.is_contacted}, synchronize_session=False)
    )
    db.commit()
    return {"status": "updated", "updated_count": updated_count}


# ---------------------------------------------------------------------------
# trackdashboard: weekly note
# ---------------------------------------------------------------------------

@app.post("/trackdashboard/weekly-note")
async def save_weekly_note(
    request: Request,
    content: str = Form(...),
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)
    from datetime import date
    week_num = date.today().isocalendar()[1]
    year = date.today().year

    note = (
        db.query(models.WeeklyNote)
        .filter(models.WeeklyNote.week_number == week_num, models.WeeklyNote.year == year)
        .first()
    )
    if note:
        note.content = content
    else:
        note = models.WeeklyNote(week_number=week_num, year=year, content=content, is_published=False)
        db.add(note)
    db.commit()
    return {"status": "saved"}


@app.post("/trackdashboard/weekly-note/publish")
async def publish_weekly_note(request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    from datetime import date
    week_num = date.today().isocalendar()[1]
    year = date.today().year

    note = (
        db.query(models.WeeklyNote)
        .filter(models.WeeklyNote.week_number == week_num, models.WeeklyNote.year == year)
        .first()
    )
    if not note:
        raise HTTPException(status_code=404, detail="No note found for this week. Save it first.")
    note.is_published = True
    note.published_at = datetime.now()
    db.commit()
    return {"status": "published"}


# ---------------------------------------------------------------------------
# trackdashboard: section visibility toggle
# ---------------------------------------------------------------------------

@app.post("/trackdashboard/sections/{section_key}/toggle")
async def toggle_section(section_key: str, request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    section = (
        db.query(models.DashboardSection)
        .filter(models.DashboardSection.section_key == section_key)
        .first()
    )
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
    section.is_visible = not section.is_visible
    section.last_updated = datetime.now()
    db.commit()
    return {"status": "toggled", "is_visible": section.is_visible}


# ---------------------------------------------------------------------------
# trackdashboard: user management
# ---------------------------------------------------------------------------

@app.post("/trackdashboard/users/create")
async def create_dashboard_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(default="sales"),
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)
    existing = db.query(models.DashboardUser).filter(models.DashboardUser.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user = models.DashboardUser(
        username=username,
        hashed_password=_hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    return {"status": "created", "username": username, "role": role}


@app.post("/trackdashboard/users/{user_id}/toggle")
async def toggle_dashboard_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    user = db.query(models.DashboardUser).filter(models.DashboardUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = not user.is_active
    db.commit()
    return {"status": "toggled", "is_active": user.is_active}


@app.delete("/trackdashboard/users/{user_id}")
async def delete_dashboard_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    user = db.query(models.DashboardUser).filter(models.DashboardUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# trackdashboard: seed default sections on first run
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def seed_default_sections():
    """
    Creates default dashboard sections if they don't exist yet.
    All start hidden (is_visible=False). Admin enables them from trackdashboard.
    """
    db = SessionLocal()
    try:
        default_sections = [
            ("hot_leads", "Hot Leads"),
            ("top_questions", "أكثر الأسئلة"),
            ("unanswered_questions", "أسئلة بلا رد"),
            ("peak_hours", "أوقات الذروة"),
            ("weekly_note", "الملاحظة الأسبوعية"),
            ("repeated_visitors", "الزوار المتكررون"),
        ]
        for key, name in default_sections:
            exists = db.query(models.DashboardSection).filter(
                models.DashboardSection.section_key == key
            ).first()
            if not exists:
                db.add(models.DashboardSection(section_key=key, section_name=name, is_visible=False))
        db.commit()
    finally:
        db.close()
# ---------------------------------------------------------------------------
# Report requests — sales submits, admin sees in trackdashboard
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def run_migrations():
    from sqlalchemy import text
    columns = [
        ("chat_logs",  "is_unanswered", "BOOLEAN DEFAULT 0"),
        ("chat_logs",  "category",  "VARCHAR"),
        ("chat_logs",  "topic",     "VARCHAR"),
        ("leads",      "is_approved", "BOOLEAN DEFAULT 0"),
        ("leads",      "approved_at", "DATETIME"),
        ("leads",      "admin_note",  "TEXT"),
        ("leads",      "session_summary", "TEXT"),

        ("leads",      "is_registered", "VARCHAR"),
        ("leads",      "city", "VARCHAR"),
        ("leads",      "question_count", "INTEGER"),
        ("leads", "is_contacted", "BOOLEAN DEFAULT 0"),

        ("uploaded_reports", "request_id",    "INTEGER"),
        ("uploaded_reports", "report_period", "VARCHAR"),
        ("uploaded_reports", "period_label",  "VARCHAR"),
        ("weekly_notes",     "published_at",  "DATETIME"),
        ("chat_logs",  "retrieved_context", "TEXT"),

    ]
    added = 0
    with engine.connect() as conn:
        # Ask SQLite what each table already has, so an up-to-date schema is a
        # no-op instead of 17 caught exceptions printed on every boot.
        existing: dict[str, set[str]] = {}
        for table in {t for t, _, _ in columns}:
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            existing[table] = {r[1] for r in rows}

        for table, col, col_type in columns:
            if not existing[table]:
                print(f"--- [MIGRATE] ! {table}: table missing, skipped ---")
                continue
            if col in existing[table]:
                continue
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                conn.commit()
                existing[table].add(col)
                added += 1
                print(f"--- [MIGRATE] ✓ {table}.{col} added ---")
            except Exception as e:
                print(f"--- [MIGRATE] ✗ {table}.{col} FAILED: {e} ---")

    print(f"--- [MIGRATE] schema up to date ({added} column(s) added) ---")

@app.post("/dashboard/request-report")
async def request_report(
    request: Request,
    report_type: str = Form(...),
    note: str = Form(default=""),
    week_num: int = Form(default=None),   
    year: int = Form(default=None),    
    db: Session = Depends(get_db),
):
    username, role = get_dashboard_user(request)
    from datetime import date
    # Use the week the user is currently viewing
    if not week_num:
        week_num = date.today().isocalendar()[1]
    if not year:
        year = date.today().year

    req = models.ReportRequest(
        requested_by=username,
        report_type=report_type,
        week_number=week_num,
        year=year,
        note=note,
        is_seen=False,
    )
    db.add(req)
    db.commit()
    return {"status": "sent"}


@app.get("/trackdashboard/notifications")
async def get_notifications(request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    unseen = (
        db.query(models.ReportRequest)
        .filter(models.ReportRequest.is_seen == False)
        .order_by(models.ReportRequest.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "requested_by": r.requested_by,
            "report_type": r.report_type,
            "week_number": r.week_number,
            "note": r.note,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for r in unseen
    ]


@app.post("/trackdashboard/notifications/{req_id}/seen")
async def mark_notification_seen(
    req_id: int, request: Request, db: Session = Depends(get_db)
):
    get_trackdashboard_user(request)
    req = db.query(models.ReportRequest).filter(models.ReportRequest.id == req_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Not found")
    req.is_seen = True
    req.seen_at = datetime.now()
    db.commit()
    return {"status": "seen"}


# ---------------------------------------------------------------------------
# Weekly data — sales navigates between past weeks
# ---------------------------------------------------------------------------

@app.get("/dashboard/week/{week_num}/{year}")
async def get_week_data(
    week_num: int,
    year: int,
    request: Request,
    db: Session = Depends(get_db),
):
    get_dashboard_user(request)
    
    from datetime import datetime, timedelta
    from collections import Counter
    

    # 1. Calculate the exact start and end dates for the requested week
    try:
        start_of_week = datetime.fromisocalendar(year, week_num, 1)
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
    except ValueError:
        # Fallback in case of invalid week/year sent from frontend
        return {"error": "Invalid week or year"}

    # 2. Fetch ONLY approved leads for the specific week directly from DB (Much faster)
    week_leads = (
        db.query(models.Lead)
        .filter(
            models.Lead.timestamp >= start_of_week,
            models.Lead.timestamp <= end_of_week
        )
        .all()
    )

    # 3. Get top questions ONLY from the sessions of these approved leads
    session_ids = [l.session_id for l in week_leads]
    top_questions = []
    
    if session_ids:
        # Fetch only the query text to save memory and increase speed
        week_queries = (
            db.query(models.ChatLog.user_query)
            .filter(
                models.ChatLog.session_id.in_(session_ids),
                models.ChatLog.timestamp >= start_of_week,
                models.ChatLog.timestamp <= end_of_week
            )
            .all()
        )
        
        # Count and format the questions
        counter = Counter(q[0] for q in week_queries if q[0])
        top_questions = [
            {"query": q, "count": c}
            for q, c in counter.most_common(25)
        ]

    # 4. Fetch the weekly note
    note = (
        db.query(models.WeeklyNote)
        .filter(
            models.WeeklyNote.week_number == week_num,
            models.WeeklyNote.year == year,
            models.WeeklyNote.is_published == True,
        )
        .first()
    )
    
    # 5. Fetch the peak report
    peak_report = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "peak_hours",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year        == year,
        )
        .first()
    )

    # Get hot and warm leads for the table 
    hot_leads_list = [l for l in week_leads if l.lead_status == 'hot']
    warm_leads_list = [l for l in week_leads if l.lead_status == 'warm']
    visible_leads = (hot_leads_list + warm_leads_list)

    # Format the leads for the JSON response
    visible_leads_data = [
        {
            "id": l.id,
            "question_count": l.question_count,
            "status": l.lead_status
        }
        for l in visible_leads
    ]
    leads_report = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "leads",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year        == year,
        )
        .order_by(models.UploadedReport.uploaded_at.desc())
        .first()
    )

    # 6. Return the exact JSON structure expected by loadWeek() in Javascript
    return {
        "week_num":      week_num,
        "year":          year,
        "week_label":    _week_label(year, week_num),
        "total_leads":   len(week_leads),
        "hot_leads":     sum(1 for l in week_leads if l.lead_status == "hot"),
        "warm_leads":    sum(1 for l in week_leads if l.lead_status == "warm"),
        "max_questions": max((l.question_count for l in week_leads), default=0),
        "weekly_note":   note.content if note else None,
        "top_questions": top_questions,
        "peak_report_id":    peak_report.id if peak_report else None,
        "peak_report_label": peak_report.period_label if peak_report else None,
        "peak_report_time":  peak_report.uploaded_at.strftime("%Y-%m-%d %H:%M") if peak_report else None,
        "visible_leads": visible_leads_data,
        "leads_report": {
            "id": leads_report.id
        } if leads_report else None,

    }
# ---------------------------------------------------------------------------
# Export leads report as HTML (for sales to download and share with team)
# ---------------------------------------------------------------------------
from typing import Optional, List
from fastapi.responses import HTMLResponse

@app.get("/trackdashboard/export-leads-html")
async def export_leads_html(
    request: Request,
    week: Optional[int] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)

    from datetime import datetime, timedelta, date
    
    # 1. Default to current week and year if not provided
    if not week: week = date.today().isocalendar()[1]
    if not year: year = date.today().year

    # 2. Calculate the exact start and end dates for the target week
    try:
        start_of_week = datetime.fromisocalendar(year, week, 1)
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
    except ValueError:
        # Fallback just in case
        return HTMLResponse(content="Invalid week or year format", status_code=400)

    # 3. Fast DB-level filtering (Only fetch leads for this specific week)
    leads = (
        db.query(models.Lead)
        .filter(
            models.Lead.timestamp >= start_of_week,
            models.Lead.timestamp <= end_of_week
        )
        .order_by(models.Lead.timestamp.desc()) # رتبهم من الأحدث للأقدم عشان التقرير يبقى شيك
        .all()
    )

    context = {
        "request":      request,
        "leads":        leads,
        "generated_at": datetime.now(),
        "week_num":     week,
        "year":         year,
        "week_label":   _week_label(year, week),
    }

    # Render template and return as downloadable file
    html_content = templates.get_template("leads_report.html").render(context)
    return HTMLResponse(
        content=html_content,
        headers={
            "Content-Disposition": f"attachment; filename=leads_week{week}_{year}.html"
        },
    )

#--------------------------------------------------------------------------
# View leads report page in browser (sales can also see it online without downloading)
#--------------------------------------------------------------------------
@app.get("/dashboard/leads-report", response_class=HTMLResponse)
async def view_leads_report_page(
    request: Request,
    db: Session = Depends(get_db),
):
    get_dashboard_user(request)
    from datetime import datetime
    from collections import defaultdict


    leads = db.query(models.Lead).order_by(models.Lead.timestamp.desc()).all()
    session_ids = [lead.session_id for lead in leads]


    questions_map = defaultdict(list)
    conversations_map = defaultdict(list)
    if session_ids:
        logs = (
            db.query(models.ChatLog.session_id, models.ChatLog.user_query, models.ChatLog.bot_answer)
            .filter(models.ChatLog.session_id.in_(session_ids))
            .order_by(models.ChatLog.timestamp.asc())
            .all()
        )
        for log in logs:
            if log.user_query:
                questions_map[log.session_id].append(log.user_query)
                conversations_map[log.session_id].append({
                    "q": log.user_query,
                    "a": log.bot_answer or "",
                })

    context = {
        "request": request,
        "leads": leads,
        "questions_map": questions_map,
        "conversations_map": conversations_map,
        "generated_at": datetime.now(),
    }

    return templates.TemplateResponse("leads_report.html", context)
# ---------------------------------------------------------------------------
# Upload HTML report from trackdashboard
# ---------------------------------------------------------------------------

@app.post("/trackdashboard/notifications/{req_id}/upload-report")
async def upload_report_for_request(
    req_id: int,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)

    req = db.query(models.ReportRequest).filter(
        models.ReportRequest.id == req_id
    ).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    content      = await file.read()
    html_content = content.decode("utf-8", errors="ignore")

    # Determine period label
    if req.report_type in ("peak_hours", "repeated_visitors"):
        period_label, report_period = _build_period_label(req.report_type, file.filename, req)
    else:
        period_label  = f"أسبوع {req.week_number} / {req.year}"
        report_period = req.report_type

    # Replace if already uploaded for same request
    existing = db.query(models.UploadedReport).filter(
        models.UploadedReport.request_id == req_id
    ).first()
    if existing:
        existing.html_content  = html_content
        existing.filename      = file.filename
        existing.uploaded_at   = datetime.now()
        existing.period_label  = period_label
        existing.report_period = report_period
    else:
        db.add(models.UploadedReport(
            request_id    = req_id,
            report_type   = req.report_type,
            report_period = report_period,
            period_label  = period_label,
            filename      = file.filename,
            html_content  = html_content,
            week_number   = req.week_number,
            year          = req.year,
        ))

    req.is_seen = True
    req.seen_at = datetime.now()
    db.commit()
    return {"status": "uploaded"}


@app.get("/view-report/{report_id}", response_class=HTMLResponse)
async def view_report_by_id(
    report_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    get_dashboard_user(request)
    report = db.query(models.UploadedReport).filter(
        models.UploadedReport.id == report_id
    ).first()
    if not report:
        return HTMLResponse("<h3 style='text-align:center;margin-top:50px;color:#aaa;'>لا يوجد تقرير</h3>")
    return HTMLResponse(content=report.html_content)

# ---------------------------------------------------------------------------
# Peak hours reports can be viewed for past weeks, so separate route with report_id
# ---------------------------------------------------------------------------
@app.get("/dashboard/peak-report/{report_id}", response_class=HTMLResponse)
async def view_peak_report(
    report_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    get_dashboard_user(request)
    report = db.query(models.UploadedReport).filter(
        models.UploadedReport.id == report_id
    ).first()
    if not report:
        return HTMLResponse("""
            <div style='text-align:center;margin-top:80px;font-family:sans-serif;color:#aaa;'>
                <div style='font-size:40px;margin-bottom:12px;'>⏳</div>
                <h3>التقرير لم يُرفع بعد</h3>
            </div>
        """)
    return HTMLResponse(content=report.html_content)

@app.get("/dashboard/peak-reports/{week_num}/{year}")
async def get_peak_reports(
    week_num: int,
    year: int,
    request: Request,
    db: Session = Depends(get_db),
):
    get_dashboard_user(request)
    reports = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "peak_hours",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year        == year,
        )
        .order_by(models.UploadedReport.uploaded_at.desc())
        .all()
    )
    return [
        {
            "id":           r.id,
            "period_label": r.period_label or f"أسبوع {week_num} / {year}",
            "uploaded_at":  str(r.uploaded_at)[:16],
        }
        for r in reports
    ]
# ---------------------------------------------------------------------------
# Question categories management (trackdashboard)
# ---------------------------------------------------------------------------

@app.post("/trackdashboard/categories/create")
async def create_category(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)
    existing = db.query(models.QuestionCategory).filter(
        models.QuestionCategory.name == name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Category already exists")
    cat = models.QuestionCategory(name=name, description=description, is_visible=False)
    db.add(cat)
    db.commit()
    db.refresh(cat)
    return {"status": "created", "id": cat.id, "name": cat.name}


@app.post("/trackdashboard/categories/{cat_id}/toggle")
async def toggle_category(cat_id: int, request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    cat = db.query(models.QuestionCategory).filter(
        models.QuestionCategory.id == cat_id
    ).first()
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    cat.is_visible = not cat.is_visible
    db.commit()
    return {"status": "toggled", "is_visible": cat.is_visible}


@app.delete("/trackdashboard/categories/{cat_id}")
async def delete_category(cat_id: int, request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    cat = db.query(models.QuestionCategory).filter(
        models.QuestionCategory.id == cat_id
    ).first()
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    db.delete(cat)
    db.commit()
    return {"status": "deleted"}


@app.get("/api/categories/questions")
async def get_category_questions(request: Request,week: int = None,year: int = None, db: Session = Depends(get_db)):
    """
    Returns visible categories with their top questions.
    Called by dashboard.html on load.
    """
    get_dashboard_user(request)
    categories = db.query(models.QuestionCategory).filter(
        models.QuestionCategory.is_visible == True
    ).all()

    result = []
    for cat in categories:
        from datetime import datetime
        if week and year:
            try:
                from datetime import timedelta
                start_of_week = datetime.fromisocalendar(year, week, 1).replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_week   = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
                q_filter = (
                    models.ChatLog.category == cat.name,
                    models.ChatLog.timestamp >= start_of_week,
                    models.ChatLog.timestamp <= end_of_week,
                )
            except ValueError:
                q_filter = (models.ChatLog.category == cat.name,)
        else:
            q_filter = (models.ChatLog.category == cat.name,)

        questions = (
            db.query(models.ChatLog.user_query, func.count(models.ChatLog.user_query).label("cnt"))
            .filter(*q_filter)
            .group_by(models.ChatLog.user_query)
            .order_by(func.count(models.ChatLog.user_query).desc())
            .limit(10)
            .all()
        )
        result.append({
            "id":        cat.id,
            "name":      cat.name,
            "questions": [{"query": q.user_query, "count": q.cnt} for q in questions],
        })

    return result


@app.get("/trackdashboard/categories/data")
async def get_all_categories(request: Request, db: Session = Depends(get_db)):
    """Returns all categories for trackdashboard management."""
    get_trackdashboard_user(request)
    cats = db.query(models.QuestionCategory).order_by(models.QuestionCategory.created_at).all()
    return [
        {
            "id":          c.id,
            "name":        c.name,
            "description": c.description,
            "is_visible":  c.is_visible,
        }
        for c in cats
    ]
#=======================================================================================
# to make html export of questions + answers for a category, so sales can share with team
#========================================================================================
@app.get("/trackdashboard/categories/{cat_id}/questions")
async def get_category_questions_full(
    cat_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)
    cat = db.query(models.QuestionCategory).filter(
        models.QuestionCategory.id == cat_id
    ).first()
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    # Get all logs in this category
    logs = (
        db.query(models.ChatLog)
        .filter(models.ChatLog.category == cat.name)
        .order_by(models.ChatLog.timestamp.desc())
        .all()
    )

    # Get phone numbers linked to sessions
    session_ids = list(set(log.session_id for log in logs))
    leads = (
        db.query(models.Lead)
        .filter(models.Lead.session_id.in_(session_ids))
        .all()
    )
    phone_map = {l.session_id: l.phone_number for l in leads}

    # Group questions by phone number
    contacts = {}
    no_phone = []

    for log in logs:
        phone = phone_map.get(log.session_id)
        qa = {
            "user_query": log.user_query,
            "bot_answer": log.bot_answer,
            "timestamp":  log.timestamp.strftime("%Y-%m-%d %H:%M"),
        }
        if phone:
            if phone not in contacts:
                contacts[phone] = {
                    "phone":     phone,
                    "questions": [],
                    "first_seen": log.timestamp.strftime("%Y-%m-%d %H:%M"),
                }
            contacts[phone]["questions"].append(qa)
        else:
            no_phone.append(qa)

    return {
        "category":  cat.name,
        "contacts":  list(contacts.values()),
        "no_phone":  no_phone,
    }


# ---------------------------------------------------------------------------
# Peak hours report — weekly
# ---------------------------------------------------------------------------
@app.get("/trackdashboard/peak-hours/weekly")
async def peak_hours_weekly(
    request: Request,
    week: int = None,
    year: int = None,
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)

    from datetime import date, timedelta
    from collections import defaultdict

    today = date.today()
    if not week: week = today.isocalendar()[1]
    if not year: year = today.year

    start_of_week, end_of_week = _week_bounds(year, week)
    week_logs = db.query(models.ChatLog).filter(
        models.ChatLog.timestamp >= start_of_week,
        models.ChatLog.timestamp <= end_of_week,
    ).all()

    day_counts  = defaultdict(int)
    hour_counts = defaultdict(int)
    for log in week_logs:
        day_counts[log.timestamp.weekday()] += 1
        hour_counts[log.timestamp.hour]     += 1

    peak_day  = max(day_counts, key=day_counts.get) if day_counts else 0
    peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 0

    import json

    context = {
        "request":         request,
        "report_type":     "weekly",
        "week_num":        week,
        "year":            year,
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "peak_day":        _DAYS_AR[peak_day] if day_counts else "—",
        "peak_hour":       f"{peak_hour:02d}:00 — {peak_hour+1:02d}:00",
        "total_sessions":  len(set(l.session_id for l in week_logs)),
        "total_questions": len(week_logs),
        "days_labels":     json.dumps([d for d in _DAYS_AR]),
        "days_data":       json.dumps([day_counts.get(i, 0) for i in range(7)]),
        "hours_labels":    json.dumps([f"{h:02d}:00" for h in range(24)]),
        "hours_data":      json.dumps([hour_counts.get(i, 0) for i in range(24)]),
        "extra_labels":    "null",
        "extra_data":      "null",
        "extra_title":     "",
    }

    html = templates.get_template("peak_hours_report.html").render(context)
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f"attachment; filename=peak_weekly_{week}_{year}.html"},
    )


# ---------------------------------------------------------------------------
# Peak hours report — monthly
# ---------------------------------------------------------------------------
@app.get("/trackdashboard/peak-hours/monthly")
async def peak_hours_monthly(
    request: Request,
    month: int = None,
    year: int = None,
    db: Session = Depends(get_db),
):
    get_trackdashboard_user(request)

    from datetime import date
    from collections import defaultdict

    today = date.today()
    if not month: month = today.month
    if not year:  year  = today.year

    start_of_month, end_of_month = _month_bounds(year, month)
    month_logs = db.query(models.ChatLog).filter(
        models.ChatLog.timestamp >= start_of_month,
        models.ChatLog.timestamp <= end_of_month,
    ).all()

    day_counts  = defaultdict(int)
    hour_counts = defaultdict(int)
    week_counts = defaultdict(int)

    for log in month_logs:
        day_counts[log.timestamp.weekday()]              += 1
        hour_counts[log.timestamp.hour]                  += 1
        week_counts[log.timestamp.isocalendar()[1]]      += 1

    peak_day  = max(day_counts,  key=day_counts.get)  if day_counts  else 0
    peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 0

    sorted_weeks = dict(sorted(week_counts.items()))

    import json

    context = {
        "request":         request,
        "report_type":     "monthly",
        "week_num":        month,
        "year":            year,
        "month_name":      _MONTHS_AR[month - 1],
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "peak_day":        _DAYS_AR[peak_day] if day_counts else "—",
        "peak_hour":       f"{peak_hour:02d}:00 — {peak_hour+1:02d}:00",
        "total_sessions":  len(set(l.session_id for l in month_logs)),
        "total_questions": len(month_logs),
        "days_labels":     json.dumps([d for d in _DAYS_AR]),
        "days_data":       json.dumps([day_counts.get(i, 0) for i in range(7)]),
        "hours_labels":    json.dumps([f"{h:02d}:00" for h in range(24)]),
        "hours_data":      json.dumps([hour_counts.get(i, 0) for i in range(24)]),
        "extra_labels":    json.dumps([f"أسبوع {w}" for w in sorted_weeks.keys()]),
        "extra_data":      json.dumps(list(sorted_weeks.values())),
        "extra_title":     "📈 توزيع النشاط على أسابيع الشهر",
    }
    html = templates.get_template("peak_hours_report.html").render(context)
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f"attachment; filename=peak_monthly_{month}_{year}.html"},
    )
@app.get("/api/topics/questions")
async def get_topic_questions(
    request: Request,
    week: int = None,
    year: int = None,
    db: Session = Depends(get_db)
):
    get_dashboard_user(request)

    from datetime import date
    from collections import Counter

    if not week: week = date.today().isocalendar()[1]
    if not year: year = date.today().year

    start_of_week, end_of_week = _week_bounds(year, week)
    week_logs = (
        db.query(models.ChatLog)
        .filter(
            models.ChatLog.topic != None,
            models.ChatLog.timestamp >= start_of_week,
            models.ChatLog.timestamp <= end_of_week,
        )
        .all()
    )

    counter = Counter(log.topic for log in week_logs)

    return [
        {"topic": topic, "count": count}
        for topic, count in counter.most_common(10)
    ]

# to identify repeated visitors by phone number, and show their questions across visits/sessions
@app.get("/trackdashboard/repeated-visitors")
async def get_repeated_visitors(request: Request, db: Session = Depends(get_db)):
    get_trackdashboard_user(request)
    from collections import defaultdict

    # Get all leads with phone numbers
    leads = (
        db.query(models.Lead)
        .filter(models.Lead.phone_number != None)
        .order_by(models.Lead.timestamp.asc())
        .all()
    )

    # Group by phone number
    phone_groups = defaultdict(list)
    for lead in leads:
        phone_groups[lead.phone_number].append(lead)

    # Keep only repeated visitors
    repeated = {
        phone: visits
        for phone, visits in phone_groups.items()
        if len(visits) > 1
    }

    result = []
    for phone, visits in repeated.items():
        # Get all questions for each visit
        visits_data = []
        for visit in visits:
            logs = (
                db.query(models.ChatLog)
                .filter(models.ChatLog.session_id == visit.session_id)
                .order_by(models.ChatLog.timestamp.asc())
                .all()
            )
            visits_data.append({
                "session_id":    visit.session_id,
                "timestamp":     visit.timestamp.strftime("%Y-%m-%d %H:%M"),
                "question_count": visit.question_count,
                "asked_price":   visit.asked_about_price,
                "asked_reg":     visit.asked_about_registration,
                "lead_status":   visit.lead_status,
                "questions": [
                    {"q": l.user_query, "a": l.bot_answer}
                    for l in logs
                ],
            })

        result.append({
            "phone":        phone,
            "visit_count":  len(visits),
            "asked_price_ever": any(v["asked_price"] for v in visits_data),
            "asked_reg_ever":   any(v["asked_reg"]   for v in visits_data),
            "visits":       visits_data,
        })

    # Sort by visit count
    result.sort(key=lambda x: x["visit_count"], reverse=True)
    return result

#to get the latest repeated visitors report for a specific week,in lasr week and next week 
@app.get("/dashboard/repeated-reports/{week_num}/{year}")
async def get_repeated_report(
    week_num: int,
    year: int,
    request: Request,
    db: Session = Depends(get_db),
):
    get_dashboard_user(request)
    reports = (
        db.query(models.UploadedReport)
        .filter(
            models.UploadedReport.report_type == "repeated_visitors",
            models.UploadedReport.week_number == week_num,
            models.UploadedReport.year        == year,
        )
        .order_by(models.UploadedReport.uploaded_at.desc())
        .all()
    )
    return {
        "reports": [
            {
                "id":           r.id,
                "period_label": r.period_label or f"أسبوع {week_num} / {year}",
                "uploaded_at":  r.uploaded_at.strftime("%Y-%m-%d %H:%M"),
            }
            for r in reports
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
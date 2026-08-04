# All security and session-related logic will be placed here
import os
import secrets
import hashlib
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
load_dotenv()

ADMIN_USER = os.getenv("ADMIN_USER")
ADMIN_PASS = os.getenv("ADMIN_PASS")
DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET", "change-me-in-production")
security = HTTPBasic()

# ---------------------------------------------------------------------------
# Admin auth (existing HTTPBasic for /admin/* routes)
# ---------------------------------------------------------------------------
def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _make_session_token(username: str, role: str) -> str:
    payload = f"{username}|{role}"
    sig = hashlib.sha256(f"{payload}{DASHBOARD_SECRET}".encode()).hexdigest()
    return f"{payload}|{sig}"


def _verify_session_token(token: str):
    """Returns (username, role) or raises HTTPException."""
    try:
        parts = token.split("|")
        username, role, sig = parts[0], parts[1], parts[2]
        expected = hashlib.sha256(f"{username}|{role}{DASHBOARD_SECRET}".encode()).hexdigest()
        if not secrets.compare_digest(sig, expected):
            raise ValueError("bad signature")
        return username, role
    except Exception:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")


def get_dashboard_user(request: Request):
    token = request.cookies.get("dashboard_session")
    if not token:
        raise HTTPException(status_code=302, headers={"Location": "/dashboard/login"})
    return _verify_session_token(token)


def get_trackdashboard_user(request: Request):
    """Only admin role can access trackdashboard."""
    username, role = get_dashboard_user(request)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return username, role


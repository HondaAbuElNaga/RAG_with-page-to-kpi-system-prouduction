from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path  # 1. استيراد مكتبة المسارات
import os
from dotenv import load_dotenv  # <--- إضافة

load_dotenv()  # <--- تشغيل

BASE_DIR = Path(__file__).resolve().parent
DB_PATH_ENV = os.getenv("DB_PATH")

if DB_PATH_ENV:
    # لو إحنا على السيرفر (Docker)، استخدم المسار الآمن
    # المسار النسبي بيتحسب من مجلد الكود مش من مكان تشغيل الأمر
    DB_FILE = Path(DB_PATH_ENV)
    if not DB_FILE.is_absolute():
        DB_FILE = BASE_DIR / DB_FILE
    DB_FILE = DB_FILE.resolve()
    print(f"--- [DB INFO] Using Persistent Volume at: {DB_FILE} ---")
else:
    # لو إحنا شغالين Local على جهازك، استخدم المسار العادي
    # data/ is the canonical location: it is what the EFS volume mounts over
    # in ECS and what the compose bind mount maps to.
    DB_FILE = BASE_DIR / "data" / "kpi_data.db"
    print(f"--- [DB INFO] Using Local File at: {DB_FILE} ---")

# Ensure parent directory exists
DB_FILE.parent.mkdir(parents=True, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

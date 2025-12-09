from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path  # 1. استيراد مكتبة المسارات
import os
from dotenv import load_dotenv  # <--- إضافة

load_dotenv()  # <--- تشغيل

DB_PATH_ENV = os.getenv("DB_PATH")

if DB_PATH_ENV:
    # لو إحنا على السيرفر (Docker)، استخدم المسار الآمن
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH_ENV}"
    print(f"--- [DB INFO] Using Persistent Volume at: {DB_PATH_ENV} ---")
else:
    # لو إحنا شغالين Local على جهازك، استخدم المسار العادي
    BASE_DIR = Path(__file__).resolve().parent
    DB_FILE = BASE_DIR / "kpi_data.db"
    # Ensure parent directory exists
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"
    print(f"--- [DB INFO] Using Local File at: {DB_FILE} ---")
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

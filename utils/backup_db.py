# backup_db.py
import os
import shutil
from datetime import datetime

DB_PATH = "database/users.db"
BACKUP_DIR = "backup"

def backup_db():
    # Tạo thư mục backup nếu chưa có
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Tạo tên file backup kèm timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"users_backup_{timestamp}.db")
    
    # Copy file database
    shutil.copy2(DB_PATH, backup_path)
    
    print(f"✅ Backup thành công: {backup_path}")

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        backup_db()
    else:
        print("❌ Không tìm thấy file database để backup!")

# email_sender.py
import os
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
#from pathlib import Path

# Load biến môi trường từ file .env
load_dotenv()

# # Load biến môi trường từ thư mục gốc dự án (1 cấp trên)
# env_path = Path(__file__).resolve().parent.parent / ".env"
# load_dotenv(dotenv_path=env_path)

def send_report_via_email(to_email, subject, body, attachments=None):
    print("SMTP_SERVER:", os.getenv("SMTP_SERVER"))
    print("SMTP_PORT:", os.getenv("SMTP_PORT"))
    print("SMTP_USERNAME:", os.getenv("SMTP_USERNAME"))
    # print("SMTP_PASSWORD:", os.getenv("SMTP_PASSWORD"))

    # Đọc cấu hình từ file .env
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    username = os.getenv("SMTP_USERNAME")   
    password = os.getenv("SMTP_PASSWORD")   


    if not all([smtp_server, smtp_port, username, password]):
        raise ValueError("❌ Thiếu thông tin SMTP trong file .env")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = to_email
    msg.set_content(body)

    # Đính kèm file PDF (nếu có)
    if attachments:
        for file_path in attachments:
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(file_path)
                    msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)
            except Exception as e:
                raise RuntimeError(f"❌ Không thể đính kèm file {file_path}: {e}")

    # Gửi email
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
            smtp.login(username, password)
            smtp.send_message(msg)
    except Exception as e:
        raise RuntimeError(f"❌ Lỗi khi gửi email: {e}")

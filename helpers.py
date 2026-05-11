import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import wandb

from config import CONFIG


def send_mails(receiver_email, subject, body):
    try:
        host = CONFIG.get("HOST")
        port = 465
        login = CONFIG.get("EMAIL_ID")
        password = CONFIG.get("PASSWORD")

        to = f"To: {receiver_email}"
        from_whom = login
        subject = subject
        body = body

        message = MIMEMultipart()
        message["To"] = to
        message["From"] = from_whom
        message["Subject"] = subject
        message["Bcc"] = ""
        message.attach(MIMEText(body, "html"))

        text_to_send = message.as_string()

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(login, password)
            server.sendmail(login, to, text_to_send)

        return True  # email was sent
    except:
        return False  # email was not sent


# Function to initialize W&B run
def initialize_wandb(project_name, run_name, resume=False):
    key = CONFIG.get("WANDB_KEY", None)
    if key:
        wandb.login(key=key)

        wandb.init(
            project=project_name,
            name=run_name,
            resume="allow" if resume else None  # "allow" enables resuming if a matching run exists
        )
    else:
        print("\nYou did not provide the WANDB Key")
        sys.exit(1)
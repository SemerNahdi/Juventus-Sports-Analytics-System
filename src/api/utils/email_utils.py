import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from ..config import SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, BASE_URL


def send_analysis_email(to_email: str, job_id: str, player_id: int, video_url: str) -> None:
    """Send summary email to user with results link."""
    if not SMTP_USER or not SMTP_PASSWORD:
        print(f"[Email Bypassed] Missing SMTP credentials. Email would have gone to {to_email}")
        return

    to_email = to_email.strip()

    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = to_email
        msg['Subject'] = f"Mitus AI: Analysis Complete - Player #{player_id}"

        dashboard_url = f"{BASE_URL}/dashboard.html?job_id={job_id}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; border: 1px solid #eee; padding: 20px; border-radius: 10px;">
                <h1 style="color: #00f0ff; background: #06070a; padding: 20px; border-radius: 10px; text-align: center; margin: 0;">Mitus AI</h1>
                <h3 style="text-align: center; color: #555; margin-top: 20px;">Analysis Report Finalized</h3>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                <p>Hello coach,</p>
                <p>Biomechanical analysis for <strong>Player #{player_id}</strong> ready for review.</p>
                <div style="margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #00f0ff;">
                    <p style="margin: 0;"><strong>Job ID:</strong> {job_id}</p>
                    <p style="margin: 5px 0 0 0;"><strong>Status:</strong> Success</p>
                </div>
                <p>Access full interactive dashboard and downloadable reports:</p>
                <div style="text-align: center; margin-top: 30px;">
                    <a href="{dashboard_url}" style="display: inline-block; padding: 15px 35px; background: #00f0ff; color: #000; text-decoration: none; border-radius: 8px; font-weight: 900; letter-spacing: 1px; text-transform: uppercase; box-shadow: 0 4px 15px rgba(0,240,255,0.3);">View Results Dashboard</a>
                </div>
                <p style="margin-top: 30px; font-size: 0.8rem; color: #888;">Annotated video: <a href="{video_url}">{video_url}</a></p>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="font-size: 0.7rem; color: #aaa; text-align: center;">Automated notification from Mitus AI Sports Analytics System.</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"[Email Sent] Analysis report for {job_id} sent to {to_email}")
    except Exception as e:
        print(f"[Email Error] Failed to send to {to_email}: {e}")
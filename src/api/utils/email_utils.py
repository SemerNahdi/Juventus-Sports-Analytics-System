import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from ..config import SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, BASE_URL


def send_analysis_email(to_email: str, job_id: str, player_id: int, video_url: str, risk_score: float = 0.0) -> None:
    """Send summary email to user with results link and risk score."""
    if not SMTP_USER or not SMTP_PASSWORD:
        print(f"[{job_id[:8]}] [Email Bypassed] Missing SMTP credentials in .env. Email would have gone to {to_email}")
        return

    to_email = to_email.strip()
    risk_label = "LOW"
    risk_color = "#10b981"
    if risk_score > 70:
        risk_label = "HIGH"
        risk_color = "#ef4444"
    elif risk_score > 40:
        risk_label = "MEDIUM"
        risk_color = "#f59e0b"

    try:
        msg = MIMEMultipart()
        msg['From'] = f"Mitus AI <{SMTP_USER}>"
        msg['To'] = to_email
        msg['Subject'] = f"Analysis Complete: Player #{player_id} (Risk: {risk_label})"

        dashboard_url = f"{BASE_URL}/dashboard.html?job_id={job_id}"

        body = f"""
        <html>
        <body style="font-family: 'Inter', Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f4f7f9; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border: 1px solid #e1e8ed; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="background: #090a0f; padding: 30px; text-align: center;">
                    <h1 style="color: #00f0ff; margin: 0; font-size: 28px; letter-spacing: 2px;">MITUS AI</h1>
                    <p style="color: #94a3b8; margin: 5px 0 0 0; text-transform: uppercase; font-size: 12px; letter-spacing: 1px;">Pro Sports Analytics</p>
                </div>
                
                <div style="padding: 40px;">
                    <h2 style="margin-top: 0; color: #1e293b;">Analysis Report Ready</h2>
                    <p>Biomechanical analysis for <strong>Player #{player_id}</strong> has been finalized and is ready for clinical review.</p>
                    
                    <div style="margin: 30px 0; padding: 20px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #00f0ff;">
                        <table style="width: 100%;">
                            <tr>
                                <td style="padding-bottom: 10px; color: #64748b; font-size: 14px;">Risk Assessment</td>
                                <td style="padding-bottom: 10px; text-align: right; font-weight: 800; color: {risk_color}; font-size: 18px;">{risk_label} ({risk_score:.1f}/100)</td>
                            </tr>
                            <tr>
                                <td style="color: #64748b; font-size: 14px;">Analysis ID</td>
                                <td style="text-align: right; font-family: monospace; color: #1e293b;">{job_id[:8]}</td>
                            </tr>
                        </table>
                    </div>

                    <p style="color: #475569;">The AI engine has processed the movement patterns and identified key performance indicators and potential injury risks.</p>
                    
                    <div style="text-align: center; margin: 40px 0;">
                        <a href="{dashboard_url}" style="display: inline-block; padding: 16px 40px; background: #00f0ff; color: #090a0f; text-decoration: none; border-radius: 8px; font-weight: 800; text-transform: uppercase; font-size: 14px; letter-spacing: 1px;">Open Interactive Dashboard</a>
                    </div>
                    
                    <p style="font-size: 13px; color: #94a3b8; text-align: center;">
                        Can't click the button? Copy and paste this link:<br>
                        <a href="{dashboard_url}" style="color: #0099ff;">{dashboard_url}</a>
                    </p>
                </div>
                
                <div style="background: #f8fafc; padding: 20px; text-align: center; border-top: 1px solid #e1e8ed;">
                    <p style="font-size: 11px; color: #94a3b8; margin: 0;">&copy; 2026 Mitus AI. All rights reserved.</p>
                    <p style="font-size: 11px; color: #cbd5e1; margin: 5px 0 0 0;">This is an automated notification. Please do not reply.</p>
                </div>
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
        print(f"[{job_id[:8]}] [Email Sent] Analysis report for {job_id} sent to {to_email} via {SMTP_USER}")
    except Exception as e:
        print(f"[{job_id[:8]}] [Email Error] Failed to send to {to_email}: {e}")
        raise e
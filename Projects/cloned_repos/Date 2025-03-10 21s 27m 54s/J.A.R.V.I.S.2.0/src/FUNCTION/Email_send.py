import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from src.FUNCTION.get_env import load_variable
from DATA.email_schema import email_prompts 
from src.BRAIN.text_to_info import send_to_ai 

def initate_email(subject:str, email_content:str) -> None:
    smtp_server = "smtp.gmail.com"
    port = 587
    password = load_variable("Password_email")
    receiver_email = load_variable("Reciever_email")
    sender_email = load_variable("Sender_email")
    
    html_content = f"""
    <html>
    <body>
    <p>{email_content}</p>
    </body>
    </html>"""
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject.strip()
        msg.attach(MIMEText(html_content, 'html'))
        
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls(context=context)
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print("Email sent successfully...")
    except Exception as e:
        print(f"Error: {e}")

def send_email() -> None:
    """send self automated email on gmail"""
    select_template = input("Select an email template (job, friend, meeting, doctor, leave, product): ")
    if select_template not in email_prompts:
        print("[+] Invalid template selection.")
        return 
    
    template = email_prompts[select_template]
    placeholders = {}
    
    for placeholder in template['prompt'].split('{')[1:]:
        placeholder_key = placeholder.split('}')[0]
        value = input(f"Enter value for '{placeholder_key}': ").strip()
        if not value:
            print(f"Value for '{placeholder_key}' cannot be empty.")
            return 
        placeholders[placeholder_key] = value
    
    formatted_prompt = template['prompt'].format(**placeholders)
    
    print("----- Start prompt -----")
    print(formatted_prompt)
    print("----- End prompt -----")
    
    # Generate email content using AI
    email_prompt = "You are a professional email writer. Write an email based on the provided content in less than 20 words."
    complete_prompt = f"{email_prompt}\n{formatted_prompt}"
    response = send_to_ai(complete_prompt).strip()
    
    # Generate email subject using AI
    sub_prompt = f"Give a suitable subject for the given email: {response}. Use 3-4 words max."
    subject = send_to_ai(sub_prompt).strip()
    return {'subject':subject , 'content':response}

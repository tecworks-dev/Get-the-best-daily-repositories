import sys
import os
import time
import json
import signal
import random
from datetime import datetime, timezone
import dns.resolver
import requests
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from rich.console import Console
from rich.padding import Padding

def clear_screen():
    """ Clears the terminal screen before output """
    os.system("cls" if os.name == "nt" else "clear")

load_dotenv()

clear_screen()

console = Console()

ascii_banner = r"""
 ______ ____ _____  __          __   _       _         _             
|  ____|  _ \_   _| \ \        / /  | |     | |       | |            
| |__  | |_) || |    \ \  /\  / /_ _| |_ ___| |__   __| | ___   __ _  
|  __| |  _ < | |     \ \/  \/ / _` | __/ __| '_ \ / _` |/ _ \ / _` |
| |    | |_) || |_     \  /\  / (_| | || (__| | | | (_| | (_) | (_| |
|_|    |____/_____|     \/  \/ \__,_|\__\___|_| |_|\__,_|\___/ \__, |
                                                                __/ |
                .--~~,__          Catching seizure banners...  |___/  
    :-....,-------`~~'._.'       before law enforcement...
    `-,,,  ,_      ;'~U'        even realizes they exist.
     _,-' ,'`-__; '--.
    (_/'~~      ''''(;                                                    

[bold blue]FBI Watchdog v1.0.1 by [link=https://darkwebinformer.com]Dark Web Informer[/link][/bold blue]
"""

console.print(Padding(f"[bold blue]{ascii_banner}[/bold blue]", (0, 0, 0, 4)))

# Domain listt to monitor for seizure banners and DNS changes
domains = ["example.com", "example1.com", "example2.com"]

# DNS records that will be checked for changes
dnsRecords = ["A", "AAAA", "CNAME", "MX", "NS", "SOA", "TXT"]

webhook_url = os.getenv("WEBHOOK")
alert_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
alert_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# If any env variables are missing, the script will not start.
if not webhook_url or not alert_bot_token or not alert_chat_id:
    console.print(Padding(f"[red]→ Missing environment variable! You did not set a WEBHOOK, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID.[/red]", (0, 0, 0, 4)))
    exit(1)

# File to store previous DNS results
state_file = "fbi_watchdog_results.json"
previous_results = {}

# Send Telegram notification for DNS changes or seizure detection
def telegram_notify(domain, record_type, records, previous_records, seizure_capture=None):
    detected_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    previous_records = previous_records if isinstance(previous_records, list) else []
    records = records if isinstance(records, list) else []

    previous_records_formatted = "\n".join(previous_records) if previous_records else "None"
    new_records_formatted = "\n".join(records) if records else "None"

    message = (
        "⚠️ *FBI Watchdog DNS Change Detected* ⚠️\n"
        "🔗 *DarkWebInformer.com - Cyber Threat Intelligence*\n\n"
        f"*Domain:* {domain}\n"
        f"*Record Type:* {record_type}\n"
        f"*Time Detected:* {detected_time}\n\n"
        f"*Previous Records:*\n```\n{previous_records_formatted}\n```\n"
        f"*New Records:*\n```\n{new_records_formatted}\n```"
    )

    payload = {
        "chat_id": alert_chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    response = requests.post(f"https://api.telegram.org/bot{alert_bot_token}/sendMessage", data=payload)

    if response.status_code != 200:
        console.print(Padding(f"→ Telegram API Error: {response.status_code}", (0, 0, 0, 4)))

    if seizure_capture and os.path.exists(seizure_capture):
        with open(seizure_capture, "rb") as photo:
            requests.post(
                f"https://api.telegram.org/bot{alert_bot_token}/sendPhoto",
                data={"chat_id": alert_chat_id, "caption": message, "parse_mode": "Markdown"},
                files={"photo": photo}
            )

# Send Discord notification for DNS changes or seizure detection
def discord_notify(domain, recordType, dnsRecords, prevEntry, screenshotPath=None):
    detected_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    prevEntry = prevEntry if isinstance(prevEntry, list) else []

    embed_data = {
        "embeds": [
            {
                "title": "⚠️ FBI Watchdog DNS Change Detected ⚠️",
                "description": (
                    "🔗 **DarkWebInformer.com - Cyber Threat Intelligence**\n\n"
                    f"**Domain:** `{domain}`\n"
                    f"**Record Type:** `{recordType}`\n"
                    f"**Time Detected:** {detected_time}\n\n"
                    f"**Previous Records:**\n```\n{'\n'.join(prevEntry) or 'None'}\n```\n"
                    f"**New Records:**\n```\n{'\n'.join(dnsRecords) or 'None'}\n```"
                ),
                "color": 16711680
            }
        ]
    }
    response = requests.post(webhook_url, json=embed_data)

    if screenshotPath and os.path.exists(screenshotPath):
        with open(screenshotPath, "rb") as image:
            requests.post(webhook_url, files={"file": image})

def capture_seizure_image(domain):
    screenshot_filename = f"screenshots/{domain}_image.png"
    os.makedirs("screenshots", exist_ok=True)

    try:
        console.print(Padding(f"→ Capturing likely LEA seizure {domain}...", (0, 0, 0, 4)))

        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--window-size=2560,1440") # 1440p resolution because WOOF. Change this to 1920,1080 if you want
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--disable-web-security")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        try:
            driver.get(f"http://{domain}")  # Try HTTP first
            time.sleep(3)
        except Exception:
            console.print(Padding(f"→ HTTP failed, switching to HTTPS... {domain}...", (0, 0, 0, 4)))
            try:
                driver.get(f"https://{domain}")
                time.sleep(3)
            except Exception as e:
                error_message = str(e)
                if "ERR_SSL_VERSION_OR_CIPHER_MISMATCH" in error_message:
                    console.print(Padding(f"→ SSL error on {domain}: Cipher mismatch, skipping screenshot.", (0, 0, 0, 4)))
                elif "ERR_NAME_NOT_RESOLVED" in error_message:
                    console.print(Padding(f"→ Domain {domain} does not exist or is offline. Skipping screenshot.", (0, 0, 0, 4)))
                else:
                    console.print(Padding(f"→ Failed to access {domain}: {error_message}", (0, 0, 0, 4)))

                driver.save_screenshot(screenshot_filename)
                console.print(Padding(f"→ Saved error page screenshot for {domain}: {screenshot_filename}", (0, 0, 0, 4)))
                driver.quit()
                return screenshot_filename

        driver.save_screenshot(screenshot_filename)
        driver.quit()
        console.print(Padding(f"→ Seizure screenshot saved: {screenshot_filename}", (0, 0, 0, 4)))
        return screenshot_filename

    except Exception as e:
        console.print(Padding(f"→ Unable to save seizure screenshot. {domain}: {e}", (0, 0, 0, 4)))
        return None

def load_previous_results():
    global previous_results
    state_file = "fbi_watchdog_results.json"
    try:
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as file:
                previous_results = json.load(file)
        else:
            previous_results = {}
    except Exception as e:
        console.print(Padding(f"[red]→ Error loading previous results: {e}[/red]", (0, 0, 0, 4)))
        previous_results = {}

# Save DNS scan results to JSON
def save_previous_results():
    state_file = "fbi_watchdog_results.json"
    try:
        with open(state_file, "w", encoding="utf-8") as file:
            json.dump(previous_results, file, indent=4, ensure_ascii=False)
        console.print("")
        console.print(Padding(f"[bold green]→ All results have been successfully saved.[/bold green]", (0, 0, 0, 4)))
    except Exception as e:
        console.print("")
        console.print(Padding(f"[red]→ Error saving results: {e}[/red]", (0, 0, 0, 4)))

exit_flag = False

def signal_handler(sig, frame):
    global exit_flag
    if exit_flag:
        console.print("")
        console.print(Padding("[red]→ Force stopping...[/red]", (0, 0, 0, 4)))
        os._exit(1)
    exit_flag = True
    sys.stdout.write("\033[2K\r")
    sys.stdout.flush()
    console.print("")
    console.print(Padding("[red]→ Safely shutting down...[/red]", (0, 0, 0, 4)))
    save_previous_results()
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Monitor domains for DNS changes and possible seizures and send alerts when needed
def watch_dog():
    global exit_flag
    try:
        while not exit_flag:
            for i, domain in enumerate(domains, start=1):
                if exit_flag:
                    break
                console.print("")
                console.print(Padding(f"[bold green]→ {(i / len(domains)) * 100:.0f}% complete[/bold green]", (0, 0, 0, 4)))

                for record_type in dnsRecords:
                    if exit_flag:
                        break
                    console.print(Padding(f"[bold cyan]→ Scanning {record_type:<5} records for {domain[:25]:<25}[/bold cyan]", (0, 0, 0, 4)))
                    
                    # Check the DNS records for the current domain
                    try:
                        answers = dns.resolver.resolve(domain, record_type, lifetime=5)
                        records = [r.to_text() for r in answers]
                    except dns.resolver.NXDOMAIN:
                        continue
                    except dns.resolver.Timeout:
                        console.print(Padding(f"[red]→ DNS check timed out for {domain}[/red]", (0, 0, 0, 4)))
                        continue
                    except:
                        records = []

                    sorted_records = sorted(records)
                    prev_entry = previous_results.get(domain, {}).get(record_type, {"records": []})
                    prev_sorted_records = sorted(prev_entry["records"])

                    if domain not in previous_results:
                        previous_results[domain] = {}

                    previous_results[domain][record_type] = {
                        "records": sorted_records
                    }

                    if sorted_records != prev_sorted_records and not exit_flag:
                        console.print("")
                        console.print(Padding(f"→ Change detected: {domain} ({record_type})", (0, 0, 0, 4)))
                        formatted_previous = "\n".join(f"   - {entry}" for entry in prev_sorted_records) or "   - None"
                        formatted_new = "\n".join(f"   - {entry}" for entry in sorted_records) or "   - None"
                        console.print("")
                        console.print(Padding(f"[yellow]→ Previous Records:[/yellow]\n[yellow]{formatted_previous}[/yellow]", (0, 0, 0, 4)))
                        console.print("")
                        console.print(Padding(f"[green]→ New Records:[/green]\n[green]{formatted_new}[/green]", (0, 0, 0, 4)))
                        console.print("")

                        seizure_capture = None
                        if record_type == "NS" and any(ns in sorted_records for ns in ["ns1.fbi.seized.gov.", "ns2.fbi.seized.gov.", "jocelyn.ns.cloudflare.com.", "plato.ns.cloudflare.com."]):
                            console.print(Padding(f"→ Taking seizure screenshot for {domain} (FBI Seized NS Detected)", (0, 0, 0, 4)))
                            seizure_capture = capture_seizure_image(domain)

                        discord_notify(domain, record_type, sorted_records, prev_sorted_records, seizure_capture)
                        telegram_notify(domain, record_type, sorted_records, prev_sorted_records, seizure_capture)

                # Add a delay between domains
                time.sleep(random.uniform(3, 6))

            if not exit_flag:
                save_previous_results()
                console.print(Padding("[bold green]→ FBI Watchdog shift complete. Snoozing for 60 seconds...[/bold green]\n", (0, 0, 0, 4)))
                interval = 60
                time.sleep(interval) # Snooze before the next shift
                
    except KeyboardInterrupt:
        exit_flag = True
        console.print(Padding("[bold red]→ Monitoring interrupted by user. Exiting...[/bold red]", (0, 0, 0, 4)))
        save_previous_results()
        console.print(Padding("[bold green]→ FBI Watchdog Results saved successfully.[/bold green]", (0, 0, 0, 4)))
        exit(0)

    except KeyboardInterrupt:
        exit_flag = True
        console.print(Padding("[bold red]→ Monitoring interrupted by user. Exiting...[/bold red]", (0, 0, 0, 4)))
        save_previous_results()
        console.print(Padding("[bold green]→ FBI Watchdog Results saved successfully.[/bold green]", (0, 0, 0, 4)))
        exit(0)

if __name__ == "__main__":
    load_previous_results()

    console.print(Padding("[bold cyan]→ Loading previous FBI Watchdog results...[/bold cyan]", (0, 0, 0, 4)))
    time.sleep(random.uniform(0.5, 1.2))

    console.print(Padding("[bold green]→ Previous FBI Watchdog results were successfully loaded...[/bold green]", (0, 0, 0, 4)))
    time.sleep(random.uniform(1.0, 2.0))

    console.print(Padding("[bold yellow]→ FBI Watchdog is starting to sniff for seizure records...[/bold yellow]\n", (0, 0, 0, 4)))
    time.sleep(random.uniform(1.5, 2.5))
    watch_dog()
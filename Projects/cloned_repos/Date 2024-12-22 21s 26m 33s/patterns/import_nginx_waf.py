import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WAF_DIR = "waf_patterns/nginx"
NGINX_WAF_DIR = "/etc/nginx/waf/"
NGINX_CONF = "/etc/nginx/nginx.conf"
INCLUDE_STATEMENT = "include /etc/nginx/waf/*.conf;"

def copy_waf_files():
    logging.info("Copying Nginx WAF patterns...")
    os.makedirs(NGINX_WAF_DIR, exist_ok=True)
    list_of_files = os.listdir(WAF_DIR)
    for conf_file in list_of_files:
        if conf_file.endswith('.conf'):
            subprocess.run(["cp", f"{WAF_DIR}/{conf_file}", NGINX_WAF_DIR], check=True)

def update_nginx_conf():
    logging.info("Ensuring WAF patterns are included in nginx.conf...")

    with open(NGINX_CONF, "r") as f:
        config = f.read()

    if INCLUDE_STATEMENT not in config:
        logging.info("Adding WAF include to nginx.conf...")
        with open(NGINX_CONF, "a") as f:
            f.write(f"\n{INCLUDE_STATEMENT}\n")
    else:
        logging.info("WAF already included in nginx.conf.")

def reload_nginx():
    logging.info("Reloading Nginx to apply new WAF rules...")
    subprocess.run(["nginx", "-t"], check=True)
    subprocess.run(["systemctl", "reload", "nginx"], check=True)

if __name__ == "__main__":
    copy_waf_files()
    update_nginx_conf()
    reload_nginx()
    logging.info("[✔] Nginx configured with latest WAF rules.")

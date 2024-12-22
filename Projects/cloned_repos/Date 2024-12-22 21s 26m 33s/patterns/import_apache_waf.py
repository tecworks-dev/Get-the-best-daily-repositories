import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WAF_DIR = "waf_patterns/apache"
APACHE_WAF_DIR = "/etc/modsecurity.d/"
APACHE_CONF = "/etc/apache2/apache2.conf"
INCLUDE_STATEMENT = "IncludeOptional /etc/modsecurity.d/*.conf"



def copy_waf_files():
    logging.info("Copying Apache WAF patterns...")
    os.makedirs(APACHE_WAF_DIR, exist_ok=True)
    list_of_files = os.listdir(WAF_DIR)
    for conf_file in list_of_files:
        if conf_file.endswith('.conf'):
            subprocess.run(["cp", f"{WAF_DIR}/{conf_file}", APACHE_WAF_DIR], check=True)

    

def update_apache_conf():
    logging.info("Ensuring WAF patterns are included in apache2.conf...")

    with open(APACHE_CONF, "r") as f:
        config = f.read()

    if INCLUDE_STATEMENT not in config:
        logging.info("Adding WAF include to apache2.conf...")
        with open(APACHE_CONF, "a") as f:
            f.write(f"\n{INCLUDE_STATEMENT}\n")
    else:
        logging.info("WAF patterns already included in apache2.conf.")

def reload_apache():
    logging.info("Reloading Apache to apply new WAF rules...")
    subprocess.run(["apachectl", "configtest"], check=True)
    subprocess.run(["systemctl", "reload", "apache2"], check=True)

if __name__ == "__main__":
    copy_waf_files()
    update_apache_conf()
    reload_apache()
    logging.info("[✔] Apache configured with latest WAF rules.")

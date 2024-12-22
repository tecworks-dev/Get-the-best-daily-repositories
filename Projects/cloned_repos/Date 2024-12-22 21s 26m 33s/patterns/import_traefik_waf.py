import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WAF_DIR = "waf_patterns/traefik"
TRAEFIK_WAF_DIR = "/etc/traefik/waf/"
TRAEFIK_DYNAMIC_CONF = "/etc/traefik/dynamic_conf.toml"
INCLUDE_STATEMENT = '[[http.routers]]\n  rule = "PathPrefix(`/`)'

def copy_waf_files():
    logging.info("Copying Traefik WAF patterns...")

    # Ensure the target directory exists
    os.makedirs(TRAEFIK_WAF_DIR, exist_ok=True)

    # Copy middleware and bot files
    for file in ["middleware.toml", "bots.toml"]:
        src_path = os.path.join(WAF_DIR, file)
        dst_path = os.path.join(TRAEFIK_WAF_DIR, file)
        
        if os.path.exists(src_path):
            subprocess.run(["cp", src_path, dst_path], check=True)
            logging.info(f"[+] {file} copied to {TRAEFIK_WAF_DIR}")
        else:
            logging.warning(f"[!] {file} not found in {WAF_DIR}")

def update_traefik_conf():
    logging.info("Ensuring WAF patterns are referenced in dynamic_conf.toml...")

    # Create dynamic_conf.toml if it doesn't exist
    if not os.path.exists(TRAEFIK_DYNAMIC_CONF):
        with open(TRAEFIK_DYNAMIC_CONF, "w") as f:
            f.write("[http.middlewares]\n")

    # Append middleware reference if not present
    with open(TRAEFIK_DYNAMIC_CONF, "r") as f:
        config = f.read()

    if INCLUDE_STATEMENT not in config:
        logging.info("Adding WAF middleware to dynamic_conf.toml...")
        with open(TRAEFIK_DYNAMIC_CONF, "a") as f:
            f.write(
                f'\n[[http.routers]]\n'
                f'  rule = "PathPrefix(`/`)"\n'
                f'  service = "traefik"\n'
                f'  middlewares = ["bad_bot_block"]\n'
            )
    else:
        logging.info("WAF middleware already referenced in dynamic_conf.toml.")

def reload_traefik():
    logging.info("Reloading Traefik to apply new WAF rules...")
    subprocess.run(["systemctl", "reload", "traefik"], check=True)

if __name__ == "__main__":
    copy_waf_files()
    update_traefik_conf()
    reload_traefik()
    logging.info("[✔] Traefik configured with latest WAF rules.")

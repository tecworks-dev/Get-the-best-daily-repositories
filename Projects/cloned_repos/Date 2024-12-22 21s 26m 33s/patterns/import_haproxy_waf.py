import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
WAF_DIR = "waf_patterns/haproxy"
HAPROXY_WAF_DIR = "/etc/haproxy/waf/"
HAPROXY_CONF = "/etc/haproxy/haproxy.cfg"

INCLUDE_STATEMENT = "    acl bad_bot hdr_sub(User-Agent) -i waf/bots.acl"

def copy_waf_files():
    logging.info("Copying HAProxy WAF patterns...")

    # Ensure the target directory exists
    os.makedirs(HAPROXY_WAF_DIR, exist_ok=True)

    # Copy acl files
    for file in ["bots.acl", "waf.acl"]:
        src_path = os.path.join(WAF_DIR, file)
        dst_path = os.path.join(HAPROXY_WAF_DIR, file)
        
        if os.path.exists(src_path):
            subprocess.run(["cp", src_path, dst_path], check=True)
            logging.info(f"[+] {file} copied to {HAPROXY_WAF_DIR}")
        else:
            logging.warning(f"[!] {file} not found in {WAF_DIR}")

def update_haproxy_conf():
    logging.info("Ensuring WAF patterns are included in haproxy.cfg...")

    with open(HAPROXY_CONF, "r") as f:
        config = f.read()

    if INCLUDE_STATEMENT not in config:
        logging.info("Adding WAF rules to haproxy.cfg...")
        with open(HAPROXY_CONF, "a") as f:
            f.write(
                f"\n# WAF and Bot Protection\n"
                f"frontend http-in\n"
                f"    bind *:80\n"
                f"    default_backend web_backend\n"
                f"    acl bad_bot hdr_sub(User-Agent) -i waf/bots.acl\n"
                f"    acl waf_attack path_reg waf/waf.acl\n"
                f"    http-request deny if bad_bot\n"
                f"    http-request deny if waf_attack\n"
            )
    else:
        logging.info("WAF patterns already included in haproxy.cfg.")

def reload_haproxy():
    logging.info("Testing HAProxy configuration...")
    subprocess.run(["haproxy", "-c", "-f", HAPROXY_CONF], check=True)
    logging.info("Reloading HAProxy to apply new WAF rules...")
    subprocess.run(["systemctl", "reload", "haproxy"], check=True)

if __name__ == "__main__":
    copy_waf_files()
    update_haproxy_conf()
    reload_haproxy()
    logging.info("[✔] HAProxy configured with latest WAF rules.")

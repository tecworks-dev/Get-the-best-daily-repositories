import os
import json

OUTPUT_DIR = "waf_patterns/traefik/"

def load_owasp_rules(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def generate_traefik_conf(rules):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_file = os.path.join(OUTPUT_DIR, "middleware.toml")

    with open(config_file, "w") as f:
        f.write("[http.middlewares]\n")
        for rule in rules:
            f.write(f"[http.middlewares.bad_bot_block_{rule['category']}]\n")
            f.write(f"  [http.middlewares.bad_bot_block_{rule['category']}.plugin.badbot]\n")
            f.write(f"    userAgent = [\"{rule['pattern']}\"]\n")
    print(f"[+] Traefik WAF rules generated at {config_file}")

if __name__ == "__main__":
    owasp_rules = load_owasp_rules("owasp_rules.json")
    generate_traefik_conf(owasp_rules)

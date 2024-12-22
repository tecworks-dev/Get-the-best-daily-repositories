import os
import json

OUTPUT_DIR = "waf_patterns/haproxy/"

def load_owasp_rules(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def generate_haproxy_conf(rules):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_file = os.path.join(OUTPUT_DIR, "waf.acl")

    with open(config_file, "w") as f:
        f.write("# HAProxy WAF ACL rules\n")
        for rule in rules:
            f.write(f"acl block_{rule['category']} hdr_sub(User-Agent) -i {rule['pattern']}\n")
            f.write(f"http-request deny if block_{rule['category']}\n")
    print(f"[+] HAProxy WAF rules generated at {config_file}")

if __name__ == "__main__":
    owasp_rules = load_owasp_rules("owasp_rules.json")
    generate_haproxy_conf(owasp_rules)

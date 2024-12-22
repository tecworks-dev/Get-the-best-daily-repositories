import json
import os
from collections import defaultdict

# Paths
INPUT_FILE = "owasp_rules.json"
OUTPUT_DIR = "waf_patterns/nginx"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_owasp_rules(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def generate_nginx_waf(rules):
    categorized_rules = defaultdict(list)
    
    # Group rules by category
    for rule in rules:
        category = rule.get("category", "generic").lower()
        categorized_rules[category].append(rule["pattern"])

    # Convert to Nginx conf files
    for category, patterns in categorized_rules.items():
        output_file = os.path.join(OUTPUT_DIR, f"{category}.conf")
        
        with open(output_file, "w") as f:
            f.write(f"# Nginx WAF rules for {category.upper()}\n")
            f.write(f"location / {{\n")
            f.write(f"    set $attack_detected 0;\n\n")
            
            # Write rules as regex checks
            for pattern in patterns:
                f.write(f"    if ($request_uri ~* \"{pattern}\") {{\n")
                f.write(f"        set $attack_detected 1;\n")
                f.write(f"    }}\n\n")
            
            # Block the request if an attack is detected
            f.write(f"    if ($attack_detected = 1) {{\n")
            f.write(f"        return 403;\n")
            f.write(f"    }}\n")
            f.write(f"}}\n")
        
        print(f"[+] Generated {output_file} ({len(patterns)} patterns)")

if __name__ == "__main__":
    print("[*] Loading OWASP rules...")
    owasp_rules = load_owasp_rules(INPUT_FILE)

    print(f"[*] Generating Nginx WAF configs from {len(owasp_rules)} rules...")
    generate_nginx_waf(owasp_rules)
    
    print("[✔] Nginx WAF configurations generated successfully.")

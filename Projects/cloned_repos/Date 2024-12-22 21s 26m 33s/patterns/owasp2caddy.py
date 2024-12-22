import json
import os
from collections import defaultdict

# Paths
INPUT_FILE = "owasp_rules.json"
OUTPUT_DIR = "waf_patterns/caddy"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_owasp_rules(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def generate_caddy_waf(rules):
    categorized_rules = defaultdict(list)

    # Group rules by category
    for rule in rules:
        category = rule.get("category", "generic").lower()
        categorized_rules[category].append(rule["pattern"])

    # Convert to Caddy conf files
    for category, patterns in categorized_rules.items():
        output_file = os.path.join(OUTPUT_DIR, f"{category}.conf")

        with open(output_file, "w") as f:
            block_name = f"block_{category}"

            # Write Caddy WAF block
            f.write(f"@{block_name} {{\n")
            f.write(f"    path_regexp {category} \"(?i)(")

            # Join all patterns with |
            f.write("|".join(patterns))
            f.write(")\"\n}\n")

            # Respond with 403 for matched patterns
            f.write(f"respond @{block_name} 403\n")

        print(f"[+] Generated {output_file} ({len(patterns)} patterns)")

if __name__ == "__main__":
    print("[*] Loading OWASP rules...")
    owasp_rules = load_owasp_rules(INPUT_FILE)

    print(f"[*] Generating Caddy WAF configs from {len(owasp_rules)} rules...")
    generate_caddy_waf(owasp_rules)

    print("[✔] Caddy WAF configurations generated successfully.")

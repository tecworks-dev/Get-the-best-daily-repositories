import json
import os
from collections import defaultdict

# Paths
INPUT_FILE = "owasp_rules.json"
OUTPUT_DIR = "waf_patterns/apache"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_owasp_rules(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def generate_apache_waf(rules):
    categorized_rules = defaultdict(list)
    
    # Group rules by category
    for rule in rules:
        category = rule.get("category", "generic").lower()
        categorized_rules[category].append(rule["pattern"])

    # Convert to Apache conf files
    for category, patterns in categorized_rules.items():
        output_file = os.path.join(OUTPUT_DIR, f"{category}.conf")
        
        with open(output_file, "w") as f:
            f.write(f"# Apache ModSecurity rules for {category.upper()}\n")
            f.write(f"SecRuleEngine On\n\n")
            
            # Write rules as ModSecurity SecRules
            for pattern in patterns:
                rule = f"SecRule REQUEST_URI \"{pattern}\" \"id:1000,phase:1,deny,status:403,log,msg:'{category} attack detected'\"\n"
                f.write(rule)
            
            print(f"[+] Generated {output_file} ({len(patterns)} patterns)")

if __name__ == "__main__":
    print("[*] Loading OWASP rules...")
    owasp_rules = load_owasp_rules(INPUT_FILE)

    print(f"[*] Generating Apache WAF configs from {len(owasp_rules)} rules...")
    generate_apache_waf(owasp_rules)
    
    print("[✔] Apache ModSecurity configurations generated successfully.")

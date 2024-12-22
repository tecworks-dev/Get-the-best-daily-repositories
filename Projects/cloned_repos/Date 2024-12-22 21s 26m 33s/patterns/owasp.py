import requests
import re
import json
import logging
from typing import List, Dict

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OWASP_CRS_BASE_URL = "https://api.github.com/repos/coreruleset/coreruleset/contents/rules"
GITHUB_REF = "v4.0"
GITHUB_REPO_URL = "https://api.github.com/repos/coreruleset/coreruleset"


def fetch_rule_files() -> List[str]:
    """
    Fetches a list of rule files from the OWASP Core Rule Set GitHub repository.
    It attempts to match a specific tag, and falls back to latest.

    Returns:
    List[str]: A list of rule file names (e.g., 'REQUEST-901-INITIALIZATION.conf').
    """
    logging.info("Fetching available rule files from GitHub...")
    # Step 1: Fetch all tags
    ref_url = f"{GITHUB_REPO_URL}/git/refs/tags"
    try:
        ref_response = requests.get(ref_url)
        ref_response.raise_for_status()
        ref_data = ref_response.json()
        available_refs = [ref['ref'] for ref in ref_data]
        logging.debug(f"Available refs: {available_refs}")
    except requests.RequestException as e:
        logging.error(f"Failed to fetch tags from {ref_url}. Reason: {e}")
        return []

    # Step 2: Find the closest matching tag
    matched_ref = next((ref for ref in available_refs if ref.endswith(f"{GITHUB_REF}.0")), None)

    if matched_ref:
        ref_sha = next(ref['object']['sha'] for ref in ref_data if ref['ref'] == matched_ref)
        logging.info(f"Found exact match for {GITHUB_REF}: {matched_ref}")
    else:
        # Fallback to latest tag
        latest_ref = ref_data[-1]
        ref_sha = latest_ref['object']['sha']
        logging.warning(f"{GITHUB_REF} not found. Using latest: {latest_ref['ref']}")

    logging.info(f"Using ref SHA: {ref_sha}")

    # Step 3: Fetch rule files using the selected SHA
    rules_url = f"{OWASP_CRS_BASE_URL}?ref={ref_sha}"
    try:
        rules_response = requests.get(rules_url)
        rules_response.raise_for_status()
        files = [item['name'] for item in rules_response.json() if item['name'].endswith('.conf')]
        logging.info(f"Found {len(files)} rule files.")
        return files
    except requests.RequestException as e:
        logging.error(f"Failed to fetch rule files from {rules_url}. Reason: {e}")
        return []


def fetch_owasp_rules(rule_files: List[str]) -> List[Dict[str, str]]:
    """
    Fetches SecRule patterns from OWASP CRS files, categorizes them, and returns a list of dictionaries.

    Parameters:
    rule_files (List[str]): A list of rule file names (e.g., ['REQUEST-901-INITIALIZATION.conf',...])

    Returns:
    List[Dict[str,str]]: A list of dictionaries, each containing a pattern and its category.
                          e.g. [{'category': 'INITIALIZATION', 'pattern':'...'},...]
    """
    logging.info("Fetching OWASP rules...")
    base_url = f"https://raw.githubusercontent.com/coreruleset/coreruleset/{GITHUB_REF}.0/rules/"
    rules = []

    for file in rule_files:
        logging.info(f"Fetching {file}...")
        try:
            response = requests.get(base_url + file)
            response.raise_for_status()
            raw_text = response.text
            sec_rules = re.findall(r'SecRule.*?"(.*?)"', raw_text, re.DOTALL)
            for rule in sec_rules:
                pattern = rule.strip().replace("\\", "")
                category = file.split('-')[-1].replace('.conf', '')
                if pattern:
                    rules.append({"category": category, "pattern": pattern})
        except requests.RequestException as e:
            logging.error(f"Failed to fetch or process {file}. Reason: {e}")
    logging.info(f"{len(rules)} rules fetched.")
    return rules


def save_as_json(rules: List[Dict[str, str]], output_file: str) -> None:
    """
    Saves the extracted rules as a JSON file.

    Parameters:
    rules (List[Dict[str, str]]): The list of extracted rules.
    output_file (str): The path of the output JSON file.
    """
    logging.info(f"Saving rules to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(rules, f, indent=4)
        logging.info(f"Rules saved successfully to {output_file}.")
    except IOError as e:
        logging.error(f"Failed to save rules to {output_file}. Reason: {e}")


if __name__ == "__main__":
    rule_files = fetch_rule_files()
    if rule_files:
        rules = fetch_owasp_rules(rule_files)
        if rules:
            save_as_json(rules, "owasp_rules.json")
    else:
        logging.error("Failed to fetch rule files. Exiting.")

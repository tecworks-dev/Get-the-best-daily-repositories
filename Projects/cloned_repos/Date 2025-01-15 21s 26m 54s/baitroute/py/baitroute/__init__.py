import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import yaml

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    path: str
    method: str
    remote_addr: str
    headers: Dict[str, str]
    body: Optional[str] = None

AlertHandler = Callable[[Alert], None]

class BaitRoute:
    """Initialize the BaitRoute with rules from the specified directory.
    
    Args:
        rules_dir (str): Path to directory containing YAML rule files
        selected_rules (Optional[List[str]]): List of rule names to load. If None, all rules are loaded.
    """
    def __init__(self, rules_dir: str, selected_rules: Optional[List[str]] = None):
        self.rules_dir = rules_dir
        self.selected_rules = selected_rules
        self.rules: List[Dict[str, Any]] = []
        self.alert_handler: Optional[AlertHandler] = None
        self._load_rules()
        banner = r''' ....
''. :   __
   \|_.'  ':       _.----._//_          Utku Sen's
  .'  .'.'-._   .'  _/ -._ \)-.----O   ___       _ _   ___          _
 '._.'.'      '--''-'._   '--..--'    | _ ) __ _(_) |_| _ \___ _  _| |_ ___
  .'.'___    /'---'. / ,-'            | _ \/ _  | |  _|   / _ \ || |  _/ -_)
_<__.-._))../ /'----'/.'_____:'.      |___/\__,_|_|\__|_|_\___/\_,_|\__\___|
:            \ ]              :  '.     
:  Acme       \\              :    '.  A web honeypot library to create decoy
:              \\__           :    .'  endpoints to detect and mislead attackers
:_______________|__]__________:...'
'''
        print()
        print(banner)
        print(f"Successfully loaded {len(self.rules)} bait endpoints")
        print()

    def _load_rules(self) -> None:
        """Load rules from YAML files in the rules directory."""
        logger.info(f"Initializing baitroute with rules directory: {self.rules_dir}")
        
        if not os.path.exists(self.rules_dir):
            raise ValueError(f"Rules directory does not exist: {self.rules_dir}")

        for root, _, files in os.walk(self.rules_dir):
            for file in files:
                if file.endswith(('.yml', '.yaml')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.rules_dir)
                    rule_name = os.path.splitext(relative_path)[0]

                    if self.selected_rules is None or rule_name in self.selected_rules:
                        with open(file_path, 'r') as f:
                            try:
                                rules = yaml.safe_load(f)
                                if isinstance(rules, list):
                                    for rule in rules:
                                        rule['name'] = rule_name
                                    self.rules.extend(rules)
                            except yaml.YAMLError as e:
                                logger.error(f"Error parsing rule file {file_path}: {e}")

        logger.info(f"Successfully loaded {len(self.rules)} bait endpoints")

    def get_matching_rule(self, path: str, method: str) -> Optional[Dict[str, Any]]:
        """Find a matching rule for the given path and method."""
        for rule in self.rules:
            if rule['path'] == path and rule['method'].upper() == method.upper():
                return rule
        return None

    def on_bait_hit(self, handler: AlertHandler) -> None:
        """Set the handler for bait endpoint hits.
        
        Args:
            handler: Callback function that takes an Alert object
        """
        self.alert_handler = handler

    def create_alert(self, path: str, method: str, remote_addr: str,
                    headers: Dict[str, str], body: Optional[str] = None) -> Alert:
        """Create an Alert object from the request details."""
        return Alert(
            path=path,
            method=method,
            remote_addr=remote_addr,
            headers=headers,
            body=body
        ) 
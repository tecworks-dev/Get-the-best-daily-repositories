import importlib
import logging
import os
import pkgutil
import subprocess
import sys
import time
import types
import typing

import pyperclip
from AppKit import NSApp
from PyQt6.QtCore import QProcess

LOG = logging.getLogger(__name__)


def get_current_process_id() -> str:
    return f"{os.getpid()}"


def get_focused_text() -> str:
    cmd_c()
    time.sleep(0.3)
    # Get clipboard content
    clipboard_content = pyperclip.paste()
    return clipboard_content.strip()


def cmd_v() -> None:
    applescript = """
    tell application "System Events"
        keystroke "v" using {command down}
    end tell
    """
    subprocess.run(["osascript", "-e", applescript])


def return_app_in_focus() -> str:
    command = """/usr/bin/osascript -e 'tell application "System Events"
                set frontApp to first application process whose frontmost is true
                return unix id of frontApp
                end tell' """
    res = subprocess.run(command, shell=True, stdin=sys.stdin, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)
    return res.stdout.strip()


def cmd_c() -> None:
    applescript = """
    tell application "System Events"
        keystroke "c" using {command down}
    end tell
    """
    subprocess.run(["osascript", "-e", applescript])


def put_app_in_focus(process_id: str) -> None:
    script = f"""tell application "System Events"
                 set frontmost of (first process whose unix id is {process_id}) to true
                 end tell"""
    QProcess.startDetached("/usr/bin/osascript", ["-e", script])


def put_this_app_in_focus() -> None:
    this_process_id = get_current_process_id()
    return put_app_in_focus(this_process_id)


def import_package_init_functions(package: types.ModuleType) -> dict[str, dict[str, typing.Callable]]:
    # List all submodules (modules and subpackages) in the package
    submodules = [module.name for module in pkgutil.iter_modules(package.__path__)]
    callables = {}

    for submodule in submodules:
        # Dynamically import the __init__.py of the subpackage
        try:
            subpackage = importlib.import_module(f"{package.__name__}.{submodule}")
            if hasattr(subpackage, "__init__"):
                # Get all callables (functions and callable objects) in the __init__.py of the subpackage
                subpackage_callables = {
                    item: getattr(subpackage, item)
                    for item in dir(subpackage)
                    if callable(getattr(subpackage, item))
                    if not item.startswith("_")
                }

                # Store callables with their names
                callables[submodule] = subpackage_callables

        except Exception as e:
            LOG.error(f"Error importing {submodule}: {e}")

    return callables


def load_all_available_functions(package: types.ModuleType) -> dict[str, dict[str, typing.Callable]]:
    return {name: functions for name, functions in import_package_init_functions(package).items() if len(functions) > 0}


def avoid_dock_macos_icon():
    NSApp.setActivationPolicy_(1)

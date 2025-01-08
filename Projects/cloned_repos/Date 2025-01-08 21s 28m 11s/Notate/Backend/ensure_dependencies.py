import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def find_python310():
    python_commands = ["python3.10", "python3"] if sys.platform != "win32" else [
        "python3.10", "py -3.10", "python"]

    for cmd in python_commands:
        try:
            result = subprocess.run(
                [cmd, "--version"], capture_output=True, text=True)
            if "Python 3.10" in result.stdout:
                return cmd
        except:
            continue
    return None


def create_venv(venv_path=None):
    if venv_path is None:
        venv_path = os.path.join(os.path.dirname(__file__), 'venv')
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        python310 = find_python310()
        if not python310:
            raise RuntimeError(
                "Python 3.10 is required but not found. Please install Python 3.10.")

        subprocess.check_call([python310, "-m", "venv", venv_path])
        print(f"Created virtual environment with {python310}")
    return venv_path


def get_venv_python(venv_path):
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    return os.path.join(venv_path, "bin", "python")


def install_package(python_path, package):
    try:
        subprocess.check_call(
            [python_path, '-m', 'pip', 'install', '--no-deps',
                '--upgrade-strategy', 'only-if-needed', package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return package, None
    except subprocess.CalledProcessError as e:
        return package, str(e)


def get_installed_packages(python_path):
    result = subprocess.run([python_path, '-m', 'pip', 'list',
                            '--format=freeze'], capture_output=True, text=True)
    return {line.split('==')[0].lower(): line.split('==')[1] for line in result.stdout.splitlines()}


def install_requirements(custom_venv_path=None):
    try:
        venv_path = create_venv(custom_venv_path)
        python_path = get_venv_python(venv_path)
        requirements_path = os.path.join(
            os.path.dirname(__file__), 'requirements.txt')

        # First handle CUDA-dependent packages
        use_cuda = os.environ.get('USE_CUDA', '0') == '1'
        sys.stdout.write("Checking CUDA availability...|0\n")
        sys.stdout.flush()

        # Install torch and torchaudio first with appropriate backend
        if use_cuda:
            subprocess.check_call(
                [python_path, '-m', 'pip', 'install', 'torch', 'torchaudio'])
            subprocess.check_call(
                [python_path, '-m', 'pip', 'install', 'openai-whisper'])
            sys.stdout.write("Installed CUDA-enabled PyTorch and Whisper|5\n")
        else:
            subprocess.check_call([python_path, '-m', 'pip', 'install',
                                   'torch', 'torchaudio',
                                   '--index-url', 'https://download.pytorch.org/whl/cpu'])
            # Install CPU version of whisper
            subprocess.check_call([python_path, '-m', 'pip', 'install',
                                   'openai-whisper',
                                   '--no-deps'])  # Don't reinstall torch
            sys.stdout.write("Installed CPU-only PyTorch and Whisper|5\n")
        sys.stdout.flush()

        # Install typing_extensions first to ensure we have the latest version
        subprocess.check_call(
            [python_path, '-m', 'pip', 'install', '--upgrade', 'typing_extensions>=4.7.0'])
        sys.stdout.write("Installed typing_extensions|10\n")
        sys.stdout.flush()

        # Now handle the rest of the requirements
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip()
                            and not line.startswith('#')
                            and not line.startswith('torch')
                            and not line.startswith('openai-whisper')
                            and not line.startswith('typing_extensions')]  # Skip packages we've already installed

        total_deps = len(requirements)
        sys.stdout.write(f"Total packages: {total_deps}|10\n")
        sys.stdout.flush()

        installed_packages = get_installed_packages(python_path)

        to_install = []
        for req in requirements:
            pkg_name = req.split('==')[0] if '==' in req else req
            if pkg_name.lower() not in installed_packages:
                to_install.append(req)

        completed_deps = total_deps - len(to_install)
        progress = 5 + (completed_deps / total_deps) * 32.5
        sys.stdout.write(f"Checked installed packages|{progress:.1f}\n")
        sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_pkg = {executor.submit(
                install_package, python_path, req): req for req in to_install}
            for future in as_completed(future_to_pkg):
                pkg = future_to_pkg[future]
                pkg_name = pkg.split('==')[0] if '==' in pkg else pkg
                result, error = future.result()
                completed_deps += 1
                progress = 37.5 + (completed_deps / total_deps) * 37.5

                if error:
                    sys.stdout.write(
                        f"Error installing {pkg_name}: {error}|{progress:.1f}\n")
                else:
                    sys.stdout.write(f"Installed {pkg_name}|{progress:.1f}\n")
                sys.stdout.flush()

        sys.stdout.write("Dependencies installed successfully!|75\n")
        sys.stdout.flush()

    except Exception as e:
        sys.stdout.write(f"Error installing dependencies: {str(e)}|0\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    custom_venv_path = sys.argv[1] if len(sys.argv) > 1 else None
    install_requirements(custom_venv_path)

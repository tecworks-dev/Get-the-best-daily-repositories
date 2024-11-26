import sys
import pathlib
import os
import urllib
import ctypes
import subprocess
import time
from urllib import request

if len(sys.argv) < 4:
    print('Usage: make_installer.py EXE_DIR OUTPUT_FILENAME VERSION')
    exit(1)

# Temporaries. Replace with  corporative repository urls.
nsis_url = 'https://deac-riga.dl.sourceforge.net/project/nsis/NSIS%203/3.02.1/nsis-3.02.1-setup.exe'
vcredistr_url = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'

temp_dir = 'temp';
res_dir = 'resources';
nsis_dir = os.path.join(os.environ["ProgramFiles(x86)"], 'NSIS')
nsis_exe_file_name = os.path.join(nsis_dir, 'makensis.exe')

def download_url(url, save_file_name):
    if not pathlib.Path(save_file_name).exists():
        print('downloading {0} to {1} ...'.format(url, save_file_name));
        request.urlretrieve(url, save_file_name)
    else:
        print('package {0} present'.format(save_file_name));

def maybe_install_nsis(url):
    if pathlib.Path(nsis_exe_file_name).exists():
        print('=== NSIS Installer found at {0}, skipping setup'
            .format(nsis_exe_file_name))
        return

    print('NSIS Installer not found. Trying to install...')

    path, file = os.path.split(os.path.normpath(url))
    save_file_name = os.path.normpath(os.path.join(temp_dir, file))

    download_url(url, save_file_name)

    if not pathlib.Path(save_file_name).exists():
        print('NSIS cannot be downloaded. Please try to install it manually')
        return

    print('Running installer in silent mode...')
    ctypes.windll.shell32.ShellExecuteW(None,
        "runas", save_file_name, '/S', None, 0)

    time.sleep(3);

    if pathlib.Path(nsis_exe_file_name).exists():
        print('+++ NSIS installed now in {0}'.format(nsis_exe_file_name))
    else:
        print('--- NSIS is still unavailable. Please try to install it '
            'manually and re-run this')

def maybe_install_nsis_exec_plugin():
    plugin_dir = 'Plugins'
    src_plugin = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), res_dir, plugin_dir );
    dst_plugin = os.path.join(nsis_dir, plugin_dir)

    nsis_exec_plugin_file_name = os.path.join(nsis_dir, plugin_dir, 'x86-ansi', 'ShellExecAsUser.dll');
    if not pathlib.Path(nsis_exec_plugin_file_name).exists():
        print('NSIS ShellExecAsUser plugin not found. Installing...')

        cmd_params = '/C xcopy "{}" "{}" /Y /S /E'.format(src_plugin, dst_plugin)
        ctypes.windll.shell32.ShellExecuteW(None, "runas", "cmd", cmd_params, None, 0);

        print('=== NSIS ShellExecAsUser installed')
    else:
        print('=== NSIS ShellExecAsUser plugin found at {0}, skipping setup'.format(nsis_exec_plugin_file_name))

def validate_path(path) :
    return os.path.abspath(os.path.expanduser(path))

if __name__ == "__main__":
    exe_dir = validate_path(sys.argv[1])
    out_filename = validate_path(sys.argv[2])
    version = sys.argv[3]

    if not pathlib.Path(exe_dir).exists() :
        print('Unable to collect data from "{0}" for it does not exists.'
            ' Aborting...'.format(exe_dir))
        exit(2)


    if not pathlib.Path(temp_dir).exists():
        os.mkdir(temp_dir)

    maybe_install_nsis(nsis_url)
    maybe_install_nsis_exec_plugin()

    download_url(vcredistr_url, os.path.join(temp_dir, 'VC_redist.x64.exe'));

    if pathlib.Path(nsis_exe_file_name).exists():
        print('Running installer building...')
        print('Collect files from "{0}"'.format(exe_dir))
        print('Creating the installer in "{0}"'.format(out_filename))
        print('Product version "{0}"'.format(out_filename))

        result = subprocess.run([nsis_exe_file_name,
            '/V4',
            '/DVERSION={0}'.format(version),
            '/DEXEDIR="{0}"'.format(exe_dir),
            '/DOUTFILENAME="{0}"'.format(out_filename),
            './installer.nsi'])
        print('Installer complited with {0}'.format(result.returncode))
        exit(result.returncode)

    exit(1)
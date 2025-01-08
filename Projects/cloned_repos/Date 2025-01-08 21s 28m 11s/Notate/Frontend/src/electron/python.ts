import { app, dialog, shell } from "electron";
import { spawn, ChildProcess, execSync, SpawnOptions } from "child_process";
import path from "path";
import { isDev } from "./util.js";
import { updateLoadingStatus } from "./loadingWindow.js";
import fs from "fs";
import log from "electron-log";
import ffmpegStatic from "ffmpeg-static";
import { generateSecret } from "./authentication/secret.js";
import { getSecret } from "./authentication/devApi.js";

log.transports.file.level = "info";
log.transports.file.resolvePathFn = () =>
  path.join(app.getPath("userData"), "logs/main.log");

let pythonProcess: ChildProcess | null = null;

async function runWithPrivileges(commands: string | string[]): Promise<void> {
  if (process.platform !== "linux") return;

  const commandArray = Array.isArray(commands) ? commands : [commands];

  try {
    // Try without privileges first
    for (const cmd of commandArray) {
      execSync(cmd);
    }
  } catch {
    log.info("Failed to run commands, requesting privileges..., ");

    const response = await dialog.showMessageBox({
      type: "question",
      buttons: ["Grant Privileges", "Cancel"],
      defaultId: 0,
      title: "Administrator Privileges Required",
      message:
        "Creating the Python environment requires administrator privileges.",
      detail:
        "This is needed to install required system dependencies and create the virtual environment. This will only be needed once.",
    });

    if (response.response === 0) {
      try {
        // Combine all commands with && to run them in sequence
        const combinedCommand = commandArray.join(" && ");
        execSync(`pkexec sh -c '${combinedCommand}'`);
      } catch (error) {
        log.error("Failed to run commands with privileges", error);
        throw new Error("Failed to run commands with elevated privileges");
      }
    } else {
      throw new Error(
        "User declined to grant administrator privileges. Cannot continue."
      );
    }
  }
}

async function ensurePythonAndVenv(backendPath: string) {
  const venvPath = path.join(backendPath, "venv");
  const pythonCommands =
    process.platform === "win32"
      ? ["python3.10", "py -3.10", "python"]
      : process.platform === "darwin"
      ? ["/opt/homebrew/bin/python3.10", "python3.10", "python3"]
      : ["python3.10", "python3"];

  let pythonCommand: string | null = null;
  let pythonVersion: string | null = null;

  for (const cmd of pythonCommands) {
    try {
      log.info(`Trying Python command: ${cmd}`);
      const version = execSync(`${cmd} --version`).toString().trim();
      log.info(`Version output: ${version}`);
      if (version.includes("3.10")) {
        pythonCommand = cmd;
        pythonVersion = version;
        log.info(`Found valid Python command: ${cmd} with version ${version}`);
        break;
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        log.info(`Failed to execute ${cmd}: ${error.message}`);
      }
      continue;
    }
  }

  if (!pythonCommand) {
    log.error("Python 3.10 is not installed or not in PATH");
    const response = await dialog.showMessageBox({
      type: "question",
      buttons: ["Install Python 3.10", "Cancel"],
      defaultId: 0,
      title: "Python 3.10 Required",
      message: "Python 3.10 is required but not found on your system.",
      detail:
        "Would you like to open the Python download page to install Python 3.10?",
    });

    if (response.response === 0) {
      // Open Python download page
      await shell.openExternal(
        "https://www.python.org/downloads/release/python-31010/"
      );
      throw new Error(
        "Please restart the application after installing Python 3.10"
      );
    } else {
      throw new Error(
        "Python 3.10 is required to run this application. Installation was cancelled."
      );
    }
  }

  log.info(`Using ${pythonVersion}`);

  const venvPython =
    process.platform === "win32"
      ? path.join(venvPath, "Scripts", "python.exe")
      : path.join(venvPath, "bin", "python");

  if (!fs.existsSync(venvPath)) {
    log.info("Creating virtual environment with Python 3.10...");

    if (process.platform === "linux") {
      try {
        // Run all commands with a single privilege prompt
        await runWithPrivileges([
          "apt-get update && apt-get install -y python3-venv python3-dev build-essential",
          `${pythonCommand} -m venv "${venvPath}"`,
          `chown -R ${process.env.USER}:${process.env.USER} "${venvPath}"`,
        ]);

        log.info("Virtual environment created successfully");
      } catch (error: unknown) {
        if (error instanceof Error) {
          log.error("Failed to create virtual environment", error);
          throw error;
        }
        throw new Error("Unknown error while creating virtual environment");
      }
    } else {
      // Original code for non-Linux systems
      try {
        execSync(`${pythonCommand} -m venv "${venvPath}"`);
        log.info("Virtual environment created successfully");
      } catch (error: unknown) {
        if (error instanceof Error) {
          log.error("Failed to create virtual environment", error);
          throw new Error("Failed to create virtual environment");
        } else {
          log.error("Unknown error in ensurePythonAndVenv", error);
          throw new Error("Unknown error in ensurePythonAndVenv");
        }
      }
    }
  }

  try {
    execSync(`"${venvPython}" -m pip install --upgrade pip`);
    log.info("Pip upgraded successfully");
  } catch (error) {
    log.error("Failed to upgrade pip", error);
    throw new Error("Failed to upgrade pip");
  }

  // Add check for NVIDIA GPU
  let hasNvidiaGpu = false;
  try {
    if (process.platform === "linux") {
      execSync("nvidia-smi");
      hasNvidiaGpu = true;
    } else if (process.platform === "win32") {
      execSync("nvidia-smi");
      hasNvidiaGpu = true;
    } else if (process.platform === "darwin") {
      // MacOS doesn't support CUDA
      hasNvidiaGpu = false;
    }
  } catch {
    log.info("No NVIDIA GPU detected, will use CPU-only packages");
    hasNvidiaGpu = false;
  }

  // Set environment variable for the Python process
  process.env.USE_CUDA = hasNvidiaGpu ? "1" : "0";

  return { venvPython, hasNvidiaGpu };
}

function extractFromAsar(sourcePath: string, destPath: string) {
  log.info(`Extracting from ${sourcePath} to ${destPath}`);
  try {
    if (!fs.existsSync(sourcePath)) {
      throw new Error(`Source path does not exist: ${sourcePath}`);
    }
    if (!fs.existsSync(destPath)) {
      log.info(`Creating directory: ${destPath}`);
      fs.mkdirSync(destPath, { recursive: true });
    }

    const files = fs.readdirSync(sourcePath);
    log.info(`Files in source: ${files.join(", ")}`);
    files.forEach((file) => {
      const fullSourcePath = path.join(sourcePath, file);
      const fullDestPath = path.join(destPath, file);

      if (fs.statSync(fullSourcePath).isDirectory()) {
        log.info(`Extracting directory: ${file}`);
        extractFromAsar(fullSourcePath, fullDestPath);
      } else {
        log.info(`Copying file: ${file}`);
        fs.copyFileSync(fullSourcePath, fullDestPath);
      }
    });
    log.info(`Extraction completed for ${sourcePath}`);
  } catch (error: unknown) {
    if (error instanceof Error) {
      log.error(`Error in extractFromAsar: ${error.message}`);
      log.error(`Stack trace: ${error.stack}`);
    } else {
      log.error(`Unknown error in extractFromAsar: ${error}`);
    }
    throw error;
  }
}

export async function startPythonServer() {
  log.info("Application starting...");
  log.info("Creating window...");
  const appPath = app.getAppPath();
  log.info(`App path: ${appPath}`);

  // Generate JWT secret before starting the server
  const jwtSecret = generateSecret();

  let backendPath;
  if (isDev()) {
    backendPath = path.join(appPath, "..", "Backend");
    log.info(`Dev mode: Backend path set to ${backendPath}`);
  } else {
    const unpackedBackendPath = path.join(appPath, "..", "Backend");
    log.info(`Prod mode: Unpacked Backend path set to ${unpackedBackendPath}`);

    if (fs.existsSync(unpackedBackendPath)) {
      backendPath = unpackedBackendPath;
      log.info(`Using unpacked Backend folder`);
    } else {
      const tempPath = path.join(app.getPath("temp"), "notate-backend");
      log.info(`Prod mode: Temp path set to ${tempPath}`);
      const asarBackendPath = path.join(appPath, "Backend");
      log.info(`Prod mode: ASAR Backend path set to ${asarBackendPath}`);
      try {
        extractFromAsar(asarBackendPath, tempPath);
        log.info(`Successfully extracted from ASAR to ${tempPath}`);
        backendPath = tempPath;
      } catch (error) {
        log.error(`Failed to extract from ASAR: ${error}`);
        throw error;
      }
    }
  }

  log.info(`Final Backend path: ${backendPath}`);
  const dependencyScript = path.join(backendPath, "ensure_dependencies.py");
  log.info(`Dependency script: ${dependencyScript}`);
  const mainScript = path.join(backendPath, "main.py");
  log.info(`Main script: ${mainScript}`);

  return new Promise((resolve, reject) => {
    let totalPackages = 0;
    let installedPackages = 0;
    ensurePythonAndVenv(backendPath)
      .then(({ venvPython, hasNvidiaGpu }) => {
        log.info(`Venv Python: ${venvPython}`);
        log.info(`CUDA enabled: ${hasNvidiaGpu}`);

        // Define spawn options with proper typing
        const spawnOptions: SpawnOptions = {
          stdio: "pipe",
          env: {
            ...process.env,
            USE_CUDA: hasNvidiaGpu ? "1" : "0",
            FFMPEG_PATH: app.isPackaged
              ? path.join(
                  process.resourcesPath,
                  "ffmpeg" + (process.platform === "win32" ? ".exe" : "")
                )
              : typeof ffmpegStatic === "string"
              ? ffmpegStatic
              : "",
            JWT_SECRET: jwtSecret,
            IS_DEV: isDev() ? "1" : "0",
            SECRET_KEY: getSecret(),
          },
        };

        // Pass the GPU status and FFmpeg path to the dependency script
        const depProcess = spawn(venvPython, [dependencyScript], spawnOptions);

        if (!depProcess.stdout || !depProcess.stderr) {
          throw new Error("Failed to create process with stdio pipes");
        }

        log.info(`Dependency process started: ${depProcess.pid}`);

        depProcess.stdout.on("data", (data: Buffer) => {
          const message = data.toString().trim();
          log.info(`Dependency process output: ${message}`);

          if (message.startsWith("Total packages:")) {
            totalPackages = parseInt(
              message.split("|")[0].split(":")[1].trim()
            );
          } else {
            const [text, progress] = message.split("|");
            if (progress) {
              updateLoadingStatus(text, parseFloat(progress));
            } else {
              updateLoadingStatus(
                text,
                (installedPackages / totalPackages) * 75
              );
            }

            if (text.includes("Installing")) {
              installedPackages++;
            }
          }
        });

        depProcess.stderr.on("data", (data: Buffer) => {
          const errorMessage = data.toString().trim();
          // Don't treat these as errors since they're actually info messages from uvicorn
          if (errorMessage.includes("INFO:")) {
            log.info(`Python info: ${errorMessage}`);
          } else {
            log.error(`Dependency check error: ${errorMessage}`);
            updateLoadingStatus(`Error: ${errorMessage}`, -1);
          }
        });

        depProcess.on("close", (code: number | null) => {
          log.info(`Dependency process closed with code ${code}`);
          if (code === 0) {
            updateLoadingStatus("Starting application server...", 80);

            // Create Python process with same options
            pythonProcess = spawn(venvPython, [mainScript], spawnOptions);

            if (
              !pythonProcess ||
              !pythonProcess.stdout ||
              !pythonProcess.stderr
            ) {
              reject(
                new Error("Failed to create Python process with stdio pipes")
              );
              return;
            }

            log.info(`Python process spawned with PID: ${pythonProcess.pid}`);

            pythonProcess.stdout.on("data", (data: Buffer) => {
              const message = data.toString().trim();
              log.info(`Python stdout: ${message}`);
              if (
                message.includes("Application startup complete.") ||
                message.includes("Uvicorn running on http://127.0.0.1:47372")
              ) {
                updateLoadingStatus("Application server ready!", 100);
                resolve(true);
              }
            });

            pythonProcess.stderr.on("data", (data: Buffer) => {
              const errorMessage = data.toString().trim();
              // Don't treat uvicorn startup messages as errors
              if (errorMessage.includes("INFO")) {
                log.info(`Python info: ${errorMessage}`);
                if (
                  errorMessage.includes("Application startup complete.") ||
                  errorMessage.includes(
                    "Uvicorn running on http://127.0.0.1:47372"
                  )
                ) {
                  updateLoadingStatus("Application server ready!", 100);
                  resolve(true);
                }
              } else {
                log.error(`Python stderr: ${errorMessage}`);
              }
            });

            pythonProcess.on("error", (error: Error) => {
              const errorMessage = `Failed to start Python server: ${error.message}`;
              log.error(errorMessage);
              updateLoadingStatus(errorMessage, -1);
              reject(error);
            });

            pythonProcess.on("close", (code: number | null) => {
              if (code !== 0) {
                const errorMessage = `Python server exited with code ${code}`;
                log.error(errorMessage);
                updateLoadingStatus(errorMessage, -1);
                reject(new Error(errorMessage));
              }
            });
          } else {
            const errorMessage = `Dependency installation failed with code ${code}`;
            log.error(errorMessage);
            updateLoadingStatus(errorMessage, -1);
            reject(new Error(errorMessage));
          }
        });
      })
      .catch((error) => {
        log.error("Failed to start Python server", error);
        reject(error);
      });
  });
}

export function stopPythonServer() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
}

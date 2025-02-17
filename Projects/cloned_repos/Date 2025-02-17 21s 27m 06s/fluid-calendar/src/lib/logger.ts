import fs from "fs";
import path from "path";

class Logger {
  private logFile: string;

  constructor() {
    const logsDir = path.join(process.cwd(), "logs");
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir);
    }
    this.logFile = path.join(
      logsDir,
      `scheduling-${new Date().toISOString().split("T")[0]}.log`
    );
  }

  log(message: string, data?: Record<string, unknown>) {
    if (
      process.env.NODE_ENV === "development" &&
      process.env.LOG_LEVEL === "debug"
    ) {
      const timestamp = new Date().toISOString();
      const logMessage = `[${timestamp}] ${message}${
        data ? "\n" + JSON.stringify(data, null, 2) : ""
      }\n`;
      fs.appendFileSync(this.logFile, logMessage);
    }
  }
}

export const logger = new Logger();

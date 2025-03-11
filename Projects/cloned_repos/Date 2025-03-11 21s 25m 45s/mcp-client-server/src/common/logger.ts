enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

class LoggerClass {
  private level: LogLevel = LogLevel.INFO;

  constructor() {
    // Set log level from environment variable
    const envLevel = process.env.LOG_LEVEL?.toUpperCase();
    if (envLevel) {
      switch (envLevel) {
        case 'DEBUG': this.level = LogLevel.DEBUG; break;
        case 'INFO': this.level = LogLevel.INFO; break;
        case 'WARN': this.level = LogLevel.WARN; break;
        case 'ERROR': this.level = LogLevel.ERROR; break;
        default:
          // Keep default level
          break;
      }
    }
  }

  init(): void {
    // Initialize logger
    this.info(`Logger initialized at level: ${LogLevel[this.level]}`);
  }

  setLevel(level: LogLevel): void {
    this.level = level;
    this.info(`Log level set to: ${LogLevel[level]}`);
  }

  debug(message: string, ...args: any[]): void {
    if (this.level <= LogLevel.DEBUG) {
      this.log('DEBUG', message, ...args);
    }
  }

  info(message: string, ...args: any[]): void {
    if (this.level <= LogLevel.INFO) {
      this.log('INFO', message, ...args);
    }
  }

  warn(message: string, ...args: any[]): void {
    if (this.level <= LogLevel.WARN) {
      this.log('WARN', message, ...args);
    }
  }

  error(message: string, ...args: any[]): void {
    if (this.level <= LogLevel.ERROR) {
      this.log('ERROR', message, ...args);
    }
  }

  private log(level: string, message: string, ...args: any[]): void {
    const timestamp = new Date().toISOString();
    const prefix = `[${timestamp}] ${level}:`;
    
    if (args.length > 0) {
      console.error(prefix, message, ...args);
    } else {
      console.error(prefix, message);
    }
  }
}

// Export singleton instance
export const Logger = new LoggerClass();
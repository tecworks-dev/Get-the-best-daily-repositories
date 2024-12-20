namespace AntiOllvm.Logging
{
    public static class Logger
    {
        public delegate void LogEvent(string message, string source);

        public static event LogEvent VerboseLog;
        public static event LogEvent InfoLog;
        public static event LogEvent WarningLog;
        public static event LogEvent ErrorLog;

        public static void VerboseNewline(string message, string source = "Program") => Verbose($"{message}{Environment.NewLine}", source);

        public static void Verbose(string message, string source = "Program")
        {
            VerboseLog(message, source);
        }
        
        public static void InfoNewline(string message, string source = "Program") => Info($"{message}{Environment.NewLine}", source);

        public static void Info(string message, string source = "Program")
        {
            InfoLog(message, source);
        }
        
        public static void WarnNewline(string message, string source = "Program") => Warn($"{message}{Environment.NewLine}", source);

        public static void Warn(string message, string source = "Program")
        {
            WarningLog(message, source);
        }
        
        public static void ErrorNewline(string message, string source = "Program") => Error($"{message}{Environment.NewLine}", source);

        public static void Error(string message, string source = "Program")
        {
            ErrorLog(message, source);
        }

        static Logger()
        {
            Init();
        }
        private static void Init()
        {
            
            Logger.InfoLog += (message, source) => Console.WriteLine($"[INFO] {source}: {message}");
            //Color the warning message 
            Logger.WarningLog += (message, source) =>
            {
                Console.ForegroundColor = ConsoleColor.DarkYellow;
                Console.WriteLine($"[WARNING] {source}: {message}");
                Console.ResetColor();
            };
            //Color the error message
            Logger.ErrorLog += (message, source) =>
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] {source}: {message}");
                Console.ResetColor();
            };

            Logger.VerboseLog += (message, source) =>
            {
                Console.ForegroundColor = ConsoleColor.DarkRed;
                Console.WriteLine($"[VERBOSE] {source}: {message}");
                Console.ResetColor();
            };
        }
    }
}
// See https://aka.ms/new-console-template for more information



// Console.WriteLine("Hello, World!");


using AntiOllvm.Helper;

namespace AntiOllvm
{
 
    internal static class Program
    {
        public static void Test()
        {
            Config config = new Config();
            config.ida_cfg_path = @"E:\RiderDemo\AntiOllvm\AntiOllvm\cfg_output_0x15e3ec.json";
            config.fix_outpath= @"E:\RiderDemo\AntiOllvm\AntiOllvm\fix.json";
            App.Init(config);
        }
        static void Main(string[] args)
        {
            if (args.Length>0)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    // 检查是否是 -s 参数
                    if (args[i] == "-s")
                    {
                        if (i + 1 < args.Length)
                        {
                            string value = args[i + 1];
                            Config config = new Config();
                            config.ida_cfg_path = value;
                            var fixJson = DirectoryHelper.GetCurrentWorkingDirectory()+"\\fix.json";
                            config.fix_outpath = fixJson;
                            App.Init(config);
                            
                        }
                        else
                        {
                            Console.WriteLine("error : -s  value is missing");
                        }
                    }
                }
            }
            else
            {
                Console.WriteLine("error : do not input -s ida_cfg_path");
            }
        }
    }
}
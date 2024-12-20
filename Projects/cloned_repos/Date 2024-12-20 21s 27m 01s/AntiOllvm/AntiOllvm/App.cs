using AntiOllvm.Analyze;
using AntiOllvm.Analyze.Type;
using AntiOllvm.Helper;
using AntiOllvm.Logging;

namespace AntiOllvm;

public class App
{
    public static void Init(Config config)
    {
        if (config == null)
        {
            Logger.ErrorNewline("config is null");
            return;
        }

    
        var readAllText = File.ReadAllText(config.ida_cfg_path);
        
      
        Simulation simulation = new(readAllText, config.fix_outpath);
        simulation.SetAnalyze(new CFFAnalyer());
        simulation.Run();
    }
}
using AntiOllvm.entity;
using AntiOllvm.Logging;

namespace AntiOllvm.Helper;


/**
 *  guess the main dispatcher
 */
public static class MainDispatchFinder
{
    /**
     * smart guess the main dispatcher block
     */
 public static Block FindMainDispatcher(Block block, RegisterContext context, List<Block> allBlocks)
 {
    //Find Cmp instruction
    List<Instruction> cmpInstructions = allBlocks.SelectMany(x => x.instructions).Where(x => x.Opcode() == OpCode.CMP).ToList();
    
    Dictionary<string, int> registerCount = new Dictionary<string, int>();
    foreach (var instruction in cmpInstructions)
    {
       var regName=  instruction.Operands()[0].registerName;
         if (!registerCount.TryAdd(regName, 1))
         {
              registerCount[regName]++;
         }
    }   
    var maxRegister = registerCount.OrderByDescending(x => x.Value).FirstOrDefault();
    //Find the first block show the max register
    foreach (var item in allBlocks)
    {
        foreach (var instruction in item.instructions)
        {
            if (instruction.Opcode() == OpCode.CMP)
            {
                if (instruction.Operands()[0].registerName == maxRegister.Key)
                {
                    return item;
                }
            }
        }
    }
    return null;
 }
}
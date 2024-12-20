using AntiOllvm.analyze;
using AntiOllvm.entity;
using AntiOllvm.Extension;
using AntiOllvm.Helper;
using AntiOllvm.Logging;

namespace AntiOllvm.Analyze.Type;

/**
 * Control Flow Flattening Analyer
 */
public class CFFAnalyer : IAnalyze
{
    private Block _findMain;

    public bool IsMainDispatcher(Block block, RegisterContext context, List<Block> allBlocks)
    {
        if (_findMain == null)
        {
            _findMain = MainDispatchFinder.FindMainDispatcher(block, context, allBlocks);
            // can't find main dispatcher   you should find in other way
           Logger.InfoNewline("Find main dispatcher " + _findMain.start_address);
        }

        return block.start_address == _findMain.start_address;
    }


    /**
     *
     */
    public bool IsChildDispatcher(Block curBlock, Block mainBlock, RegisterContext registerContext)
    {
        var isChild1 = ChildDispatchFinder.IsChildDispatch1(curBlock, mainBlock, registerContext);
        if (isChild1)
        {
            return true;
        }

        var isChild2 = ChildDispatchFinder.IsChildDispatch2(curBlock, mainBlock, registerContext);
        if (isChild2)
        {
            return true;
        }

        return false;
    }

    public bool IsRealBlock(Block block, Block mainBlock, RegisterContext context)
    {
        return true;
    }

    /**
     * Return real block has child block
     */
    public bool IsRealBlockWithDispatchNextBlock(Block block, Block mainDispatcher, RegisterContext regContext,
        Simulation simulation)
    {
        var mainRegisterName = mainDispatcher.GetMainDispatchOperandRegisterName();
        foreach (var instruction in block.instructions)
        {
            if (instruction.Opcode() == OpCode.B &&
                instruction.GetRelativeAddress() == mainDispatcher.GetStartAddress())
            {
                return true;
            }
        }
        // loc_15E604
        // LDR             X9, [SP,#0x2D0+var_2B0]
        // ADRP            X8, #qword_7289B8@PAGE
        // LDR             X8, [X8,#qword_7289B8@PAGEOFF]
        // STR             X9, [SP,#0x2D0+var_260]
        // LDR             X9, [SP,#0x2D0+var_2A8]
        // STR             X8, [SP,#0x2D0+var_238]
        // MOV             W8, #0x561D9EF8
        // STP             X19, X9, [SP,#0x2D0+var_270]
        foreach (var instruction in block.instructions)
        {
            if (instruction.Opcode() == OpCode.MOV || instruction.Opcode()==OpCode.MOVK)
            {
                var second = instruction.Operands()[1];

                if (instruction.Operands()[0].registerName == mainRegisterName &&
                    second.kind == Arm64OperandKind.Immediate)
                {
                    return true;
                }
            }
            
        }
       
        return false;
    }
}
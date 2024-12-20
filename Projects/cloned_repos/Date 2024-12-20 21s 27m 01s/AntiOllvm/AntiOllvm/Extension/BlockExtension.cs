using System.Text;
using AntiOllvm.Analyze;
using AntiOllvm.entity;
using AntiOllvm.Helper;
using AntiOllvm.Logging;

namespace AntiOllvm.Extension;

public static class BlockExtension
{
    public static List<Block> GetLinkedBlocks(this Block block, Simulation simulation)
    {
        var linkedBlocks = new List<Block>();
        foreach (var address in block.LinkBlocks)
        {
            linkedBlocks.Add(simulation.FindBlockByAddress(address));
        }

        return linkedBlocks;
    }

    public static string GetMainDispatchOperandRegisterName(this Block block)
    {
        foreach (var instruction in block.instructions)
        {
            switch (instruction.Opcode())
            {
                case OpCode.CMP:
                {
                    var operand = instruction.Operands()[0];
                    return operand.registerName;
                }
            }
        }

        return "";
    }

    /**
     *  find block is end to mainDispatch
     */
    public static bool IsEndJumpToMainDispatch(this Block block, Block mainDispath)
    {
        var ins = block.instructions[^1];
        switch (ins.Opcode())
        {
            case OpCode.B:
            {
                if (ins.Operands()[0].pcRelativeValue == mainDispath.GetStartAddress())
                {
                    return true;
                }

                return false;
            }
        }

        return false;
    }

    /**
     *  Get Control Flow Flattening CESL EXP
     */
    public static Instruction GetCFF_CSEL_Expression(this Block block, Block mainDispatch, RegisterContext regContext)
    {
        foreach (var instruction in block.instructions)
        {
            switch (instruction.Opcode())
            {
                case OpCode.CSEL:
                {
                    var mainCompareName = mainDispatch.GetMainDispatchOperandRegisterName();
                    var first = instruction.Operands()[0].registerName;
                    var second = instruction.Operands()[1].registerName;
                    var third = instruction.Operands()[2].registerName;
                    if (mainCompareName == first)
                    {
                        var secondReg = regContext.GetRegister(second);
                        var thirdReg = regContext.GetRegister(third);
                        if (secondReg.GetLongValue() != 0 && thirdReg.GetLongValue() != 0)
                        {
                            // Logger.InfoNewline("Find CFF_CSEL_Expression " + instruction);
                            return instruction;
                        }
                    }

                    break;
                }
            }
        }

        return null!;
    }

    public static bool HasCSELOpCode(this Block block)
    {
        foreach (var instruction in block.instructions)
        {
            switch (instruction.Opcode())
            {
                case OpCode.CSEL:
                {
                    return true;
                }
            }
        }

        return false;
    }

    public static void LinkToFirstRealBlock(this Block block, long firstBlockAddress)
    {
        for (int i = 0; i < block.instructions.Count; i++)
        {
            var ins = block.instructions[i];
            if (i + 1 == block.instructions.Count)
            {
                //Jump
                ins.fixmachine_code = AssemBuildHelper.BuildJump(ins.FormatOpcode(OpCode.B), firstBlockAddress);
            }
            else
            {
                ins.fixmachine_code = "NOP";
            }
        }
    }

    private static int FindOperandDispatchInstructionIndex(this Block block, Block main)
    {
        for (int i = 0; i < block.instructions.Count(); i++)
        {
            var instruction = block.instructions[i];
            if (instruction.Opcode() == OpCode.MOV)
            {
                var first = instruction.Operands()[0];
                var second = instruction.Operands()[1];
                if (first.registerName == main.GetMainDispatchOperandRegisterName() &&
                    second.kind == Arm64OperandKind.Immediate)
                {
                    var l = second.immediateValue;
                    if (l.ToString("X").Length == 8)
                    {
                        return i;
                    }
                }
            }
        }


        return -1;
    }

    public static bool CheckCESLAfterMove(this Block block, int CSELindex, Simulation simulation)
    {
        var nextIns = block.instructions[CSELindex + 1];
        if (nextIns.Opcode() == OpCode.MOVK || nextIns.Opcode() == OpCode.MOV)
        {
            var operand = nextIns.Operands()[1];
            if (operand.kind == Arm64OperandKind.Immediate)
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsJumpToSameBlock(this Block block, Block main, Simulation simulation)
    {
        var links = block.GetLinkedBlocks(simulation);
        if (links.Count == 0)
        {
            return false;
        }

        if (links.Count() == 1)
        {
            if (block.RealChilds is { Count: 1 })
            {
                var link = links[0];
                if (link.GetStartAddress() == block.RealChilds[0].GetStartAddress())
                {
                    return true;
                }

                return false;
            }
        }

        return false;
    }

    private static bool IsJumpToDispatcherButNotWithBIns(this Block block, Block main, Simulation simulation)
    {
       var ins=  block.instructions[^1];
       if (ins.Opcode()!=OpCode.B)
       {
           return true;
       }
       return false;
    }
    private static bool IsJumpToDispatcher(this Block block, Block main, Simulation simulation)
    {
        var links = block.GetLinkedBlocks(simulation);
        if (links.Count == 0)
        {
            return false;
        }

        if (links.Count() == 1)
        {
            if (block.RealChilds is { Count: 1 })
            {
                var link = links[0];
                if (link.GetStartAddress() == block.RealChilds[0].GetStartAddress())
                {
                    return false;
                }

                // it's not jump to same block it's mean jump to bransh
                return true;
            }
        }

        return false;
    }

    private static void FixJumpToDispatchButNotBIns(this Block block, Block main, Simulation simulation)
    {
        var insIndex = block.FindOperandDispatchInstructionIndex(main);
        if (insIndex != -1)
        {
            //fix address in this index after 
            var ins = block.instructions[insIndex];
            for (int i = 0; i < block.instructions.Count; i++)
            {
                var item = block.instructions[i];

                if (i > insIndex)
                {
                    // Logger.InfoNewline(" find Fix ins "+item);
                    item.address = $"0x{(item.GetAddress() - ins.InstructionSize).ToString("X")}";
                    // Logger.InfoNewline("Fix Address " + item.address);
                    item.fixmachine_byteCode = item.machine_code;
                }
            }

            //Remove dispatch instruction
            block.instructions.RemoveAt(insIndex);
            var lastIns = block.instructions[^1];

            //Add B instruction to Last instruction
            ins.fixmachine_code =
                AssemBuildHelper.BuildJump(ins.FormatOpcode(OpCode.B), block.RealChilds[0].GetStartAddress());
            ins.operands_str = $"0x{block.RealChilds[0].GetStartAddress().ToString("X")}";
            ins.mnemonic = ins.FormatOpcode(OpCode.B);
            ins.address = $"0x{(lastIns.GetAddress() + 4).ToString("X")}";
            block.instructions.Add(ins);
            //Fix Address
            if (ins.InstructionSize == 8)
            {
                block.instructions.Add(Instruction.CreateNOP($"0x{ins.GetAddress() + 4:X}"));
            }
        }
    }
    public static void FixMachineCodeNew(this Block block, Block main, Simulation simulation)
    {
        block.isFix = true;
        // if this not null  it's must be CFF_CSEL not other logic block
        if (block.HasCFF_CSEL())
        {
            var index = block.FindIndex(block.CFF_CSEL);
            if (index + 1 == block.instructions.Count - 1)
            {
                FixCSEL(block, block.CFF_CSEL, main);
                return;
            }

            if (index + 2 == block.instructions.Count - 1)
            {
                // CSEL            W8, W22, W21, LT
                // MOVK            W10, #0x186A,LSL#16
                // B               loc_15E510
                Logger.WarnNewline("===============Fix CESLAFTERMOVE==================");
                if (block.CESLAfterMoveOpreand)
                {
                    var movIns = block.instructions[index + 1];
                    if (movIns.InstructionSize == 4)
                    {
                        if (movIns.Opcode() == OpCode.MOV || movIns.Opcode() == OpCode.MOVK)
                        {
                            FixCSEL(block, block.CFF_CSEL, main);
                            //Final Fix Link
                            var link = block.RealChilds[2];
                            var linkAddress = link.GetStartAddress();
                            var lastIns = block.instructions[^1];
                            var newopcode= AssemBuildHelper.BuildJump(lastIns.FormatOpcode(OpCode.B), linkAddress);
                            Logger.InfoNewline("Final Fix CSLEAfterMove  with  " + newopcode);
                            lastIns.fixmachine_code =newopcode;
                            foreach (var VARIABLE in block.RealChilds)
                            {
                        
                                Logger.ErrorNewline(" Child is " + VARIABLE.start_address);
                            }
                            return;
                        }
                    }
                
                }
                throw new Exception(" CFF_CSEL Not support fix this machine code \n" + block);  
            }
        }
        else
        {
            //Dont have CFF_CSEL
            Logger.ErrorNewline(" FixMachineCodeNew  Dont have CFF_CSEL  IsJumpToDispatcher " +
                                IsJumpToDispatcher(block, main, simulation));
            if (IsJumpToDispatcher(block, main, simulation))
            {
                if (IsJumpToDispatcherButNotWithBIns(block,main,simulation))
                {   
                    //loc_15E604
                    // LDR             X9, [SP,#0x2D0+var_2B0]
                    // ADRP            X8, #qword_7289B8@PAGE
                    // LDR             X8, [X8,#qword_7289B8@PAGEOFF]
                    // STR             X9, [SP,#0x2D0+var_260]
                    // LDR             X9, [SP,#0x2D0+var_2A8]
                    // STR             X8, [SP,#0x2D0+var_238]
                    // MOV             W8, #0x561D9EF8
                    // STP             X19, X9, [SP,#0x2D0+var_270]
                
                    // loc_15E628
                    // CMP             W8, W23
                    // B.GT            loc_15E6C0
                    // Logger.ErrorNewline("IsJumpToDispatcherButNotWithBIns  FixMachineCodeNew  \n" + block);
                    FixJumpToDispatchButNotBIns( block, main, simulation);
                  
                   
                    return;
                }
               
                var ins = block.instructions[^1];
                if (block.RealChilds == null)
                {
                    Logger.InfoNewline("Error Block  \n" + block);
                    throw new Exception(" Block is null");
                }

                if (block.RealChilds.Count == 1)
                {
                    var nextBlock = block.RealChilds[0];
                    var nextBlockAddress = nextBlock.GetStartAddress();
                    Logger.WarnNewline("FixMachineCode  with   Jump To Dispatcher Block   NextBlock " +
                                       nextBlockAddress.ToString("X")
                                       + " Block \n" + block);
                    ins.fixmachine_code = AssemBuildHelper.BuildJump(ins.FormatOpcode(ins.Opcode()), nextBlockAddress);
                }
                else
                {
                    throw new Exception("Not support fix this machine code" + block.start_address);
                }
            }
            else
            {
               
                Logger.WarnNewline("!!!!!!!!!!!!Warning is inLine Block ? IsJumpSameBlock ? " +
                                   IsJumpToSameBlock(block, main, simulation)
                                   + " \n" + block);
            }
        }
    }


    private static void FixCSEL(Block block, Instruction csel, Block mainDispatcher)
    {
        Logger.WarnNewline("Fix CSEL  \n" + block);
        var cselIndex = block.FindIndex(csel);
        var opcode = csel.GetCSELOpCodeFix();
        var matchBlockAddress = block.RealChilds[0].GetStartAddress();
        var JumpIns = AssemBuildHelper.BuildJump(csel.FormatOpcode(opcode), matchBlockAddress);
        csel.fixmachine_code = JumpIns;
        var nextIns = block.instructions[cselIndex + 1];
        var notMatchBlockAddress = block.RealChilds[1].GetStartAddress();
        var notMatchJumpIns = AssemBuildHelper.BuildJump(nextIns.FormatOpcode(OpCode.B), notMatchBlockAddress);
        nextIns.fixmachine_code = notMatchJumpIns;
    }

    public static int FindIndex(this Block block, Instruction instruction)
    {
        for (int i = 0; i < block.instructions.Count; i++)
        {
            var item = block.instructions[i];
            if (instruction.GetAddress() == item.GetAddress())
            {
                return i;
            }
        }

        return -1;
    }

    public static bool IsChangeDispatchRegisterInRealBlock(this Block block, Block mainDispatcher)
    {
        var mainRegisterName = mainDispatcher.GetMainDispatchOperandRegisterName();
        foreach (var instruction in block.instructions)
        {
            if (instruction.Opcode() == OpCode.MOVK)
            {
                var second = instruction.Operands()[1];

                if (instruction.Operands()[0].registerName == mainRegisterName &&
                    second.kind == Arm64OperandKind.Immediate)
                {
                    return true;
                }
            }

            if (instruction.Opcode() == OpCode.MOV)
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
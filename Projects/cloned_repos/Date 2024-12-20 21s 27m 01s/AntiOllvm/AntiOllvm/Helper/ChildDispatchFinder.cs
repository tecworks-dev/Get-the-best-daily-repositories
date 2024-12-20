using AntiOllvm.entity;
using AntiOllvm.Extension;

namespace AntiOllvm.Helper;

public static class ChildDispatchFinder
{
    /**
     * Find Child when this block only 2 instruction
     * CMP W8,W9
     * B.EQ 0X100
     */
    public static bool IsChildDispatch1(Block curBlock, Block mainBlock, RegisterContext registerContext)
    {
        var operandRegName = mainBlock.GetMainDispatchOperandRegisterName();
        foreach (var instruction in curBlock.instructions)
        {
            switch (instruction.Opcode())
            {
                case OpCode.CMP:
                {
                    var operand = instruction.Operands()[0];

                    var right = instruction.Operands()[1];
                    if (right is { kind: Arm64OperandKind.Immediate })
                    {
                        return false;
                    }
                    var imm = registerContext.GetRegister(right.registerName).GetLongValue();

                    if (operandRegName == operand.registerName && imm != 0 &&
                        curBlock.instructions.Count == mainBlock.instructions.Count)
                    {
                        return true;
                    }

                    return false;
                }
            }
        }

        return false;
    }

    /**
     * Find Child in this case
     * loc_15E5A8
        MOV         W9, #0xD9210058
        CMP         W8, W9
        B.EQ        loc_15E604
     */
    public static bool IsChildDispatch2(Block curBlock, Block mainBlock, RegisterContext registerContext)
    {
        var operandRegName = mainBlock.GetMainDispatchOperandRegisterName();
        foreach (var instruction in curBlock.instructions)
        {
            switch (instruction.Opcode())
            {
                case OpCode.CMP:
                {
                    var operand = instruction.Operands()[0];

                    var right = instruction.Operands()[1];
                    if (operand.registerName == operandRegName)
                    {
                        if (right.kind == Arm64OperandKind.Register)
                        {
                            var isImmToReg = IsMoveImmediateToRegister(curBlock, mainBlock, registerContext,
                                out var registerName);
                            if (isImmToReg && registerName == right.registerName)
                            {
                                return true;
                            }
                        }
                    }

                    return false;
                }
            }
        }

        return false;
    }

    private static bool IsMoveImmediateToRegister(Block block, Block mainBlock, RegisterContext registerContext,
        out string registerName)
    {
        registerName = "";
        foreach (var instruction in block.instructions)
        {
            switch (instruction.Opcode())
            {
                case OpCode.MOV:
                {
                    var operand = instruction.Operands()[0];
                    if (operand.kind == Arm64OperandKind.Register)
                    {
                        var right = instruction.Operands()[1];
                        if (right is { kind: Arm64OperandKind.Immediate, immediateValue: 0 })
                        {
                            return false;
                        }

                        if (right.kind == Arm64OperandKind.Register)
                        {
                            return false;
                        }

                        registerName = operand.registerName;
                        return true;
                    }

                    return false;
                }
            }
        }

        return false;
    }
}
using AntiOllvm.entity;
using AntiOllvm.Logging;

namespace AntiOllvm.Extension;

public static class InstructionsExtension
{

    public static string FormatOpcode(this Instruction instruction,OpCode opCode)
    {
        switch (opCode)
        {
            case OpCode.B:
            {
                return "B";
            }
            case OpCode.B_EQ:
            {
                return "B.EQ";
            }
            case OpCode.B_LT:
            {
                return "B.LT";
            }
            case OpCode.MOVK:
            {
                return "MOVK";
            }
            case OpCode.MOV:
            {
                return "MOV";
            }
            case OpCode.B_NE:
            {
                return "B.NE";
            }
        }
        throw new Exception(" FormatOpcode not support " + opCode);
    }
    
    public static bool IsJumpToMainDispatcher(this Instruction instruction,Block mainDispatch)
    {
        var mainDispatchAddress = mainDispatch.GetStartAddress();
        var jumpAddress = instruction.GetRelativeAddress();
        if (jumpAddress == mainDispatchAddress)
        {
            return true;
        }

        return false;
    }
    public static bool IsOperandDispatchRegister(this Instruction instruction,Block mainDispatch,RegisterContext regContext)
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
                return true;
            }
        }

        return false;
    }
    public static OpCode FormatOpCode(this Instruction instruction)
    {
        var mnemonic= instruction.mnemonic;
        if (string.IsNullOrEmpty(mnemonic))
        {
            return OpCode.NONE;
        }

        return mnemonic switch
        {
            "ADRL" => OpCode.ADRL,
            "ADRP" => OpCode.ADRP,
            "LDP" => OpCode.LDP,
            "MOV" => OpCode.MOV,
            "MOVK" => OpCode.MOVK,
            "CMP" => OpCode.CMP,
            "B.LE" => OpCode.B_LE,
            "B.GT" => OpCode.B_GT,
            "B.EQ" => OpCode.B_EQ,
            "B.NE" => OpCode.B_NE,
            "BL" => OpCode.BL,
            "B" => OpCode.B,
            "SUB" => OpCode.SUB,
            "ADD" => OpCode.ADD,
            "LDUR" => OpCode.LDUR,
            "STR" => OpCode.STR,
            "STP" => OpCode.STP,
            "LDR" => OpCode.LDR,
            "STUR" => OpCode.STUR,
            "MRS" => OpCode.MRS,
            "SXTW" => OpCode.SXTW,
            "CSEL" => OpCode.CSEL,
            "LDRSW" => OpCode.LDRSW,
            "LDRB" => OpCode.LDRB,
            "EOR" => OpCode.EOR,
            "STRB" => OpCode.STRB,
            "MSUB" => OpCode.MSUB,
            "RET" => OpCode.RET,
            "CBZ" => OpCode.CBZ,
            "STRH" => OpCode.STRH,
            "FCMP" => OpCode.FCMP,
            "AND" => OpCode.AND,
            "DUP" => OpCode.DUP,
            "MOVI" => OpCode.MOVI,
            "SDIV" => OpCode.SDIV,
            "NOP"   => OpCode.NOP,
            "CMN"   => OpCode.CMN,
            "SUBS" => OpCode.SUBS,
            _ => throw new Exception("Unknown mnemonic: " + mnemonic)
        };
    }
    public static Arm64ConditionCode GetCSELCompareOpCode( this Instruction instruction)
    {
        var operand= instruction.Operands()[3];
        if (operand.kind==Arm64OperandKind.ConditionCode)
        {
            return operand.conditionCode;
        }
        throw new Exception("CSEL not support " + instruction.Opcode() +" ins "+instruction);
    
    }
    public static OpCode GetCSELOpCodeFix(this Instruction instruction)
    {
        var operand= instruction.Operands()[3];
        if (operand.kind==Arm64OperandKind.ConditionCode)
        {
            switch (operand.conditionCode)
            {
                case Arm64ConditionCode.EQ:
                {
                    return OpCode.B_EQ;
                }
                case Arm64ConditionCode.LT:
                {
                    return OpCode.B_LT;
                }
                case Arm64ConditionCode.NE:
                {
                    return OpCode.B_NE;
                }
            }
        }
        throw new Exception("CSEL not support " + instruction.Opcode() +" ins "+instruction);
    
    }
}
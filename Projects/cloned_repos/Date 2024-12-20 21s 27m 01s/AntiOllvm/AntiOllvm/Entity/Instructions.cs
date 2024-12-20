using System.Globalization;
using AntiOllvm.Extension;
using AntiOllvm.Logging;
using Newtonsoft.Json;

namespace AntiOllvm.entity;

[Serializable]
public class Instruction
{
    public string address { get; set; }
    public string mnemonic { get; set; }
    public string operands_str { get; set; }
    public string machine_code { get; set; }
    
    public string fixmachine_code { get; set; }
    public string fixmachine_byteCode { get; set; }
    
    
    public int InstructionSize => GetInstructionSize();
    
    private int GetInstructionSize()
    {
        return machine_code.Length / 2;
    }
    public long GetAddress()
    {
        return Convert.ToInt64(address.Replace("0x", ""), 16);
    }

    public long GetRelativeAddress()
    {
        return Convert.ToInt64(operands_str.Replace("loc_", ""), 16);
    }

    public OpCode Opcode() => GetOpCode();
    public Operand[] Operands() => GetOperands(operands_str);


    [JsonIgnore] private bool _isParsed = false;
    [JsonIgnore] private Operand[] cachedOperands;

    private OpCode GetOpCode()
    {
        return this.FormatOpCode();
    }

    private Operand[] GetOperands(string operandsStr)
    {
        if (_isParsed)
        {
            return cachedOperands;
        }

        string[] operands = operandsStr.Split(',');
        Operand[] result = new Operand[operands.Length];
        for (int i = 0; i < operands.Length; i++)
        {
            result[i] = ParserOperand(operands[i]);
        }

        _isParsed = true;
        cachedOperands = result;
        return result;
    }

    private bool IsRegister(string operand_str)
    {
        return operand_str.StartsWith("X") || operand_str.StartsWith("W");
    }

    private bool IsVectorRegisterElement(string operand_str)
    {
        return operand_str.StartsWith("v");
    }

    private bool IsImmediate(string operand_str)
    {
        return operand_str.StartsWith("#");
    }


    private bool IsMemory(string operand_str)
    {
        return operand_str.StartsWith("[") && operand_str.EndsWith("]");
    }

    private Operand ParserOperand(string operand_str)
    {
        var operand = new Operand();
        operand.operand_str = operand_str;

        if (IsRegister(operand_str))
        {
            operand.kind = Arm64OperandKind.Register;
            operand.registerName = operand_str;
        }
        else if (IsImmediate(operand_str))
        {
            operand.kind = Arm64OperandKind.Immediate;
            if (operand_str.Contains("0x"))
            {
                operand_str = operand_str.Replace("0x", "");
            }

            var imm = operand_str.Replace("#", "");
            operand.immediateValue = imm == "0" ? 0 : Convert.ToInt64(imm, 16);
        }
        else if (IsVectorRegisterElement(operands_str))
        {
            operand.kind = Arm64OperandKind.VectorRegisterElement;
        }


        else if (operand_str.StartsWith("[") && operand_str.EndsWith("]"))
        {
            operand.kind = Arm64OperandKind.Memory;
        }
        else if (operand_str.StartsWith("loc_"))
        {
            operand.kind = Arm64OperandKind.ImmediatePcRelative;
            operand.pcRelativeValue = Convert.ToInt64(operand_str.Replace("loc_", ""), 16);
        }
        else if (operand_str.StartsWith("LSL#"))
        {
            operand.kind = Arm64OperandKind.ShiftedRegister;
            operand.shiftValue = Convert.ToInt32(operand_str.Replace("LSL#", ""));
            operand.shiftType = Arm64ShiftType.LSL;
        }
        else
        {
            var hasConditionCode = IsConditionCode(operand_str);
            if (hasConditionCode != Arm64ConditionCode.NONE)
            {
                operand.kind = Arm64OperandKind.ConditionCode;
                operand.conditionCode = hasConditionCode;
            }
            else
            {
                operand.kind = Arm64OperandKind.None;
            }
        }

        return operand;
    }

    private Arm64ConditionCode IsConditionCode(string operandStr)
    {
        switch (operandStr)
        {
            case "EQ":
                return Arm64ConditionCode.EQ;
            case "NE":
                return Arm64ConditionCode.NE;
            case "CS":
                return Arm64ConditionCode.CS;
            case "CC":
                return Arm64ConditionCode.CC;
            case "MI":
                return Arm64ConditionCode.MI;
            case "PL":
                return Arm64ConditionCode.PL;
            case "VS":
                return Arm64ConditionCode.VS;
            case "VC":
                return Arm64ConditionCode.VC;
            case "HI":
                return Arm64ConditionCode.HI;
            case "LS":
                return Arm64ConditionCode.LS;
            case "GE":
                return Arm64ConditionCode.GE;
            case "LT":
                return Arm64ConditionCode.LT;
            case "GT":
                return Arm64ConditionCode.GT;
            case "LE":
                return Arm64ConditionCode.LE;
            case "AL":
                return Arm64ConditionCode.AL;
            case "NV":
                return Arm64ConditionCode.NV;
            default:
                return Arm64ConditionCode.NONE;
        }
    }

    public override string ToString()
    {
        return $"{mnemonic} {operands_str}";
    }

    public static Instruction CreateNOP(string address)
    {
        Instruction ins= new();
        ins.address = address;
        ins.mnemonic = "NOP";
        ins.operands_str = "";
        ins.machine_code = "D503201F";
        ins.fixmachine_code = "NOP";
        return ins;
    }
}
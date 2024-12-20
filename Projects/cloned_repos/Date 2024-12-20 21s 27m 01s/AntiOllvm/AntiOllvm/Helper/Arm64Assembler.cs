namespace AntiOllvm.Helper;

public class Arm64Assembler
{
    // 条件码映射
    static readonly Dictionary<string, int> CondCodes = new Dictionary<string, int>
    {
        { "EQ", 0b0000 },
        { "NE", 0b0001 },
        { "HS", 0b0010 },
        { "LO", 0b0011 },
        { "MI", 0b0100 },
        { "PL", 0b0101 },
        { "VS", 0b0110 },
        { "VC", 0b0111 },
        { "HI", 0b1000 },
        { "LS", 0b1001 },
        { "GE", 0b1010 },
        { "LT", 0b1011 },
        { "GT", 0b1100 },
        { "LE", 0b1101 },
        { "AL", 0b1110 }
    };

    public static string Assemble(string instruction, long curAddr)
    {
        if (instruction == "NOP")
        {
            return EncodeNop();
        }

       
        var menmonic = instruction.Split(" ")[0];
        if (menmonic.StartsWith("B."))
        {   
            int offset = 0;
            int addr;
            var address= instruction.Split(" ")[1];
            if (address.StartsWith("0x"))
            {
                addr = Convert.ToInt32(address, 16);
            }
            else
            {
                addr = Convert.ToInt32(address);
            }
            offset = addr - (int)curAddr+4;
            var cond = menmonic.Split(".")[1];
            return EncodeBCond(cond, offset);
        }
        else if (menmonic == "B")
        {
            int offset = 0;
            int addr;
            var address= instruction.Split(" ")[1];
            if (address.StartsWith("0x"))
            {
                addr = Convert.ToInt32(address, 16);
            }
            else
            {
                addr = Convert.ToInt32(address);
            }
            offset = addr - (int)curAddr+4;
            return EncodeB(offset);
        }
        return "";
    }

    static string EncodeB(int offset)
    {
        const int opcode = 0b000101;
        if (offset % 4 != 0)
            throw new ArgumentException("Offset must be a multiple of 4 for B instruction.");

        int imm26 = offset / 4;
        if (imm26 < -(1 << 25) || imm26 >= (1 << 25))
            throw new ArgumentOutOfRangeException("Offset out of range for B instruction.");

        uint imm26Masked = (uint)(imm26 & 0x03FFFFFF); // 26位
        uint machineCode = ((uint)opcode << 26) | imm26Masked;
        return machineCode.ToString("X8");
    }

    static string EncodeBCond(string cond, int offset)
    {
        const int opcode = 0b01010100000;
        if (!CondCodes.ContainsKey(cond))
            throw new ArgumentException($"Unsupported condition code: {cond}");

        if (offset % 4 != 0)
            throw new ArgumentException("Offset must be a multiple of 4 for B.cond instruction.");

        int condCode = CondCodes[cond];
        int imm19 = offset / 4;
        if (imm19 < -(1 << 18) || imm19 >= (1 << 18))
            throw new ArgumentOutOfRangeException("Offset out of range for B.cond instruction.");

        uint imm19Masked = (uint)(imm19 & 0x7FFFF); // 19位
        uint machineCode = ((uint)opcode << 21) | ((uint)condCode << 16) | imm19Masked;
        return machineCode.ToString("X8");
    }

    static string EncodeNop()
    {
        return "D503201F";
    }
}
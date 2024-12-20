using System.Text;

namespace AntiOllvm.Helper;

public static class AssemBuildHelper
{



    public static string BuildJump(string opcode, long address)
    {
        var ins = new StringBuilder();
        ins.Append(opcode);
        ins.Append(" ");
        ins.Append("0x");
        ins.Append(address.ToString("X"));
        return ins.ToString();
    }
}
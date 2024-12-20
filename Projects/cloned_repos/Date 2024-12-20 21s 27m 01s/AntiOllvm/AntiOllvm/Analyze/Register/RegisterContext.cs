using AntiOllvm.Logging;

namespace AntiOllvm;

public class RegisterContext
{
    private readonly List<Register> _registers = new();
    
    private Dictionary<string,List<Register>> _registerSnapshot = new();
    private bool N;
    private bool Z;
    private bool C;
    private bool V;

    public RegisterContext()
    {
        for (int i = 0; i < 31; i++)
        {
            _registers.Add(new Register { name = "X" + i, value = 0 });
        }

        _registers.Add(new Register { name = "SP", value = 0 });
        
    }
    public void RestoreRegisters(string key)
    {
        if (_registerSnapshot.ContainsKey(key))
        {
            var snapshot = _registerSnapshot[key];
            for (int i = 0; i < _registers.Count; i++)
            {
                _registers[i] = snapshot[i];
            }
        }
        else
        {
            throw new Exception(" Register snapshot not found");

        }
    }
    public bool SnapshotRegisters(string key)
    {
        var snapshot = new List<Register>();
        foreach (var register in _registers)
        {
            snapshot.Add((Register)register.Clone());
        }

        if (_registerSnapshot.ContainsKey(key))
        {   
            throw   new Exception("Register snapshot already exists");
            return false;
        }
        _registerSnapshot.Add(key, snapshot);
        return true;
    }
    public void LogRegisters()
    {
        foreach (var register in _registers)
        {
            Logger.InfoNewline(register.name + " = " + register.GetLongValue().ToString("X"));
        }
    }
    
    public void SetRegister(string name, object value)
    {
        long v = (long)value;
        // Logger.InfoNewline("SetRegister  " + name + " = " + v.ToString("X"));
        name = name.Replace("W", "X");
        GetRegister(name).value = value;
        
    }
    public Register GetRegister(string name)
    {
        name = name.Replace("W", "X");
        return _registers.Find(register => register.name == name) ?? throw new Exception("Register not found");
    }

    private bool IsWRegister(string operand_str)
    {
        return operand_str.StartsWith("W");
    }

    public static void CompareTest()
    {
        long left = 0x186363ed;
        long right = 0xfa639493;
       
        // 将无符号操作数转换为有符号整数，以便进行有符号比较
        int signedLeft = unchecked((int)left);
        int signedRight = unchecked((int)right);
        int signedResult = signedLeft - signedRight;

        // 设置N（Negative）标志位：结果为负
      bool  N = signedResult < 0;

        // 设置Z（Zero）标志位：结果为零
      bool  Z = (left == right);

        // 设置C（Carry）标志位：无借位（即左操作数 >= 右操作数，无符号比较）
      bool  C = left >= right;

        // 设置V（Overflow）标志位：有符号溢出
      bool  V = ((signedLeft < 0 && signedRight > 0 && signedResult > 0) ||
             (signedLeft > 0 && signedRight < 0 && signedResult < 0));

        // 记录比较过程和结果
        Logger.InfoNewline($"Comparing left (0x{left:X8}) with right (0x{right:X8})");
        Logger.InfoNewline($"Result = left - right = 0x{signedResult:X8} ({signedResult})");
        Logger.InfoNewline($"N = {N}");
        Logger.InfoNewline($"Z = {Z}");
        Logger.InfoNewline($"C = {C}");
        Logger.InfoNewline($"V = {V}");
        
    }
    /**
     * Compare two registers and set flags
     */
    public void Compare(string leftRegisterName, string rightRegisterName)
    {
        long left = GetRegister(leftRegisterName).GetLongValue();
        long right = GetRegister(rightRegisterName).GetLongValue();
        // Logger.InfoNewline("Comparing " + leftRegisterName + " = " + left +"(0x"+left.ToString("X")+")"+ " with " + rightRegisterName + " = " + right + "(0x"+right.ToString("X")+ ")");
        var result = left - right;
        N = result < 0;
        Z = result == 0;
        if (IsWRegister(leftRegisterName) && IsWRegister(rightRegisterName))
        {
            var i_left = (uint)left;
            var i_right = (uint)right;
            C = i_left >= i_right;
            V = ((i_left < 0 && i_right > 0 && result > 0) ||
                 (i_left > 0 && i_right < 0 && result < 0)); // Overflow flag
        }
        else
        {
            C = left >= right;
            V = ((left < 0 && right > 0 && result > 0) ||
                 (left > 0 && right < 0 && result < 0)); // Overflow flag
        }
    }
}
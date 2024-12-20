using AntiOllvm.entity;
using AntiOllvm.Logging;

namespace AntiOllvm;

public class Register : ICloneable
{
    public string name { get; set; }
    public object value { get; set; }

    public const long UNKNOWN_VALUE = 0;

    public long GetLongValue()
    {
        if (value is long)
        {
            IConvertible convertible = (IConvertible)value;
            return convertible.ToLong();
        }

        return UNKNOWN_VALUE;
    }

    public int GetIntValue()
    {
        try
        {
            var l = GetLongValue();
            return Convert.ToInt32(l);
        }
        catch (Exception e)
        {   long l = GetLongValue();
            var i = (int)GetLongValue();
            // Logger.InfoNewline($" {l} ({l:X}) Change To  {i}");
            return i;
        }
      
    }

    public object Clone()
    {
        return new Register
        {
            name = name,
            value = value
        };
    }
}
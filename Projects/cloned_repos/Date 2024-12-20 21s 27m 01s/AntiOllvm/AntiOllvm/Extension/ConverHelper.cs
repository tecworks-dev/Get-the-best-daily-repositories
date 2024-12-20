using System.Globalization;

namespace AntiOllvm.entity;

public static class ConverHelper
{
    public static long ToLong(this IConvertible convertible)
    {
        return convertible.GetTypeCode() switch
        {
            TypeCode.Int32 => (int)convertible,
            _ => convertible.ToInt64(CultureInfo.InvariantCulture)
        };
    }
}
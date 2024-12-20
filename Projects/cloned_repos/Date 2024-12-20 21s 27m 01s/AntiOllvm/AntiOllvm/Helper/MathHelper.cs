namespace AntiOllvm.entity;

public class MathHelper
{
    public static long CalculateMOVK(long dest, long imm, Arm64ShiftType shift, int shiftValue)
    {   
        int shiftAmount = 0;
        switch (shift)
        {
            case Arm64ShiftType.LSL:
                shiftAmount = shiftValue;
                break;
            case Arm64ShiftType.LSR:
                shiftAmount = shiftValue;
                break;
            case Arm64ShiftType.ASR:
                shiftAmount = shiftValue;
                break;
            case Arm64ShiftType.ROR:
                shiftAmount = shiftValue;
                break;
        }
        // 创建掩码，用于清除目标位段的值
        long mask = ~(0xFFFFL << shiftAmount);

        // 清除目标位段，并将立即数插入
        long result = (dest & mask) | ((imm & 0xFFFFL) << shiftAmount);

        return result;
    } 
    
}
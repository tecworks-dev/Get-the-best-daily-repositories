using AntiOllvm.entity;
using AntiOllvm.Logging;

namespace AntiOllvm.Helper;

public class ConditionJumpHelper
{
  
    public static bool Condition(OpCode opCode, Instruction lastCompareIns, RegisterContext regContext)
    {
        var left = lastCompareIns.Operands()[0];
        var right = lastCompareIns.Operands()[1];
        if (!left.registerName.StartsWith("W") || !right.registerName.StartsWith("W"))
        {
            throw new Exception(" not support register");
        }
        var leftReg = left.registerName;
        var rightReg = right.registerName;
        var leftV = regContext.GetRegister(left.registerName).GetIntValue();
        var rightV = regContext.GetRegister(right.registerName).GetIntValue();
        switch (opCode)
        {
            case OpCode.B_NE:
            {   
               Logger.InfoNewline($" B.NE   {leftReg} : {leftV}  != {rightReg} : {rightV}");
                if (leftV != rightV)
                {
                    return true;
                }

                return false;
            }
            case OpCode.B_LE:
            {   
                
              
                Logger.InfoNewline($" B.LE   {leftReg} : {leftV}  <= {rightReg} : {rightV}");
                if (leftV <= rightV)
                {
                    return true;
                }

                return false;
            }
            case OpCode.B_EQ:
            {
              
                Logger.InfoNewline($" B.EQ   {leftReg} : {leftV}  == {rightReg} : {rightV}");
                if (leftV == rightV)
                {
                    return true;
                }

                return false;
            }
            case OpCode.B_GT:
            {
                //有符号比较
                Logger.InfoNewline($" B.GT   {leftReg} : {leftV}  > {rightReg} : {rightV}");
                if (leftV > rightV)
                {
                    return true;
                }

                return false;
            }
        }

        throw new Exception("Not implemented " + opCode);
    }
}
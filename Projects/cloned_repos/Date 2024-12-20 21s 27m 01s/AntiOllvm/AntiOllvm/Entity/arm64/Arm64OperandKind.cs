namespace AntiOllvm.entity;

public enum Arm64OperandKind
{
    None,
    Register,
   
    VectorRegisterElement,
 
    Immediate,
  
    ImmediatePcRelative,
    ConditionCode,
    FloatingPointImmediate,
    ShiftedRegister,
    Memory
}
namespace AntiOllvm.entity;

public class JsonBlock
{
    public long address;
    public List<Instruction> instructions;
    public List<JsonBlock> linkBlocks;
}
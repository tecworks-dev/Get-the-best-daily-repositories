namespace AntiOllvm.entity;

public class IDACFG
{
    public string fun_start { get; set; }
    public string fun_end { get; set; }
    public List<Block> cfg { get; set; }
}
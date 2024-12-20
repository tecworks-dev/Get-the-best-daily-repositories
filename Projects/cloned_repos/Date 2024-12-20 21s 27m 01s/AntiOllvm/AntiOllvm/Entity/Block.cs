using System.Text;
using Newtonsoft.Json;

namespace AntiOllvm.entity;

[Serializable]
public class Block
{
    public string start_address { get; set; }
    public string end_address { get; set; }
    public List<Instruction> instructions { get; set; }
    
    
    public Block FixLinkBlock { get; set; }
    
    [JsonIgnore]
    public bool CESLAfterMoveOpreand { get; set; }
    [JsonIgnore]
    public bool isFind { get; set; }
    
    [JsonIgnore]
    public bool isFix { get; set; }
    public List<string> linkBlocks { get; set; }
    
    public List<Block> RealChilds { get; set; }

    [JsonIgnore] public Instruction CFF_CSEL = null!;
  
    [JsonIgnore]
    public List<long> LinkBlocks => GetLinkBlocks();

    public bool HasCFF_CSEL()
    {
        return CFF_CSEL != null;
    }
   
    public bool IsChildBlock(Block block)
    {
        return LinkBlocks.Contains(block.GetStartAddress());
    }

    private List<long> GetLinkBlocks()
    {
        List<long> result = new List<long>();
        foreach (var link in linkBlocks)
        {
            result.Add(Convert.ToInt64(link.Replace("0x", ""), 16));
        }

        return result;
    }

    public long GetStartAddress()
    {
        return Convert.ToInt64(start_address.Replace("0x", ""), 16);
    }

    public long GetEndAddress()
    {
        return Convert.ToInt64(end_address.Replace("0x", ""), 16);
    }

    public override bool Equals(object? obj)
    {
        if (obj == null)
        {
            return false;
        }

        if (obj.GetType() != this.GetType())
        {
            return false;
        }

        Block block = (Block)obj;
        return block.start_address == this.start_address && block.end_address == this.end_address;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        foreach (var VARIABLE in instructions)
        {
            sb.Append(VARIABLE.address + "   " + VARIABLE.mnemonic + " " + VARIABLE.operands_str + "\n");
        }

        return $"[Block] {start_address} \n{sb}";
    }

    public string OutJson()
    {
        return JsonHelper.ToString(this);
    }


  
}
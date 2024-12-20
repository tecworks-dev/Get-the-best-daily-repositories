using AntiOllvm.Analyze;
using AntiOllvm.entity;

namespace AntiOllvm.analyze;

public interface IAnalyze
{
    public bool IsMainDispatcher(Block block, RegisterContext context,List<Block> allBlocks);
    public bool IsChildDispatcher(Block curBlock, Block mainBlock, RegisterContext registerContext);
    public bool IsRealBlock(Block block, Block mainBlock, RegisterContext context);

    bool IsRealBlockWithDispatchNextBlock(Block block, Block mainDispatcher, RegisterContext regContext, Simulation simulation);
}
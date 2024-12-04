#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

struct ChangeTargetTriplePass : public PassInfoMixin<ChangeTargetTriplePass>
{
    // New target triple
    const std::string NewTriple = "mipsel-unknown-linux-elf";

    PreservedAnalyses run(Module &M, ModuleAnalysisManager &)
    {
        if (M.getTargetTriple() != NewTriple)
        {
            // Print the old triple for debugging
            errs() << "Changing target triple from: " << M.getTargetTriple() << " to: " << NewTriple << "\n";

            // Update the target triple
            M.setTargetTriple(NewTriple);
            return PreservedAnalyses::none();
        }

        return PreservedAnalyses::all();
    }
};

llvm::PassPluginLibraryInfo getChangeTargetTriplePassPluginInfo()
{
    return {
        LLVM_PLUGIN_API_VERSION, "ChangeTriplePass", LLVM_VERSION_STRING,
        [](PassBuilder &PB)
        {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>)
                {
                    if (Name == "change-triple")
                    {
                        MPM.addPass(ChangeTargetTriplePass());
                        return true;
                    }
                    return false;
                }
            );
        }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return getChangeTargetTriplePassPluginInfo();
}
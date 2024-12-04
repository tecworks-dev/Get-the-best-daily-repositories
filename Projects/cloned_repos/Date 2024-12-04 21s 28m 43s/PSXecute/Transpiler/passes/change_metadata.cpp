#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

struct ChangeMetadataPass : public PassInfoMixin<ChangeMetadataPass>
{
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &)
    {
        for (Function &F : M)
        {
            // Remove x86-specific function attributes
            F.removeFnAttr("target-cpu");
            F.removeFnAttr("target-features");
            F.removeFnAttr("tune-cpu");
            F.removeFnAttr("stack-protector-buffer-size");
        }

        // Adjust the data layout to match MIPS I 32-bit (mipsel-linux-unknown-elf)
        M.setDataLayout("e-m:e-i32:32-f64:64-n32:64-S128");

        errs() << "Adjusted metadata"
               << "\n";

        return PreservedAnalyses::none();
    }
};

llvm::PassPluginLibraryInfo getChangeMetadataPassPluginInfo()
{
    return {
        LLVM_PLUGIN_API_VERSION, "ChangeMetadataPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB)
        {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>)
                {
                    if (Name == "change-metadata")
                    {
                        MPM.addPass(ChangeMetadataPass());
                        return true;
                    }
                    return false;
                }
            );
        }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return getChangeMetadataPassPluginInfo();
}
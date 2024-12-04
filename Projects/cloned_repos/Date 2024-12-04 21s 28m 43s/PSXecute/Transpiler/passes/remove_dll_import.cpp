#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

struct RemoveDllImportPass : public PassInfoMixin<RemoveDllImportPass>
{
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &)
    {
        bool modified = false;

        // for all functions in module
        for (auto it = M.begin(), end = M.end(); it != end;)
        {
            Function &F = *it++;

            if (F.hasDLLImportStorageClass())
            {
                errs() << "Removing dllimport declaration: " << F.getName() << "\n";
                F.eraseFromParent();
                modified = true;
            }
        }

        // Iterate through all global variables in the module
        for (auto it = M.global_begin(), end = M.global_end(); it != end;)
        {
            GlobalVariable &GV = *it++;

            if (GV.hasDLLImportStorageClass())
            {
                errs() << "Removing dllimport global variable: " << GV.getName() << "\n";
                GV.eraseFromParent();
                modified = true;
            }
        }

        return modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

llvm::PassPluginLibraryInfo getRemoveDllImportPassPluginInfo()
{
    return {
        LLVM_PLUGIN_API_VERSION, "RemoveDllImportPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB)
        {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>)
                {
                    if (Name == "remove-dllimport")
                    {
                        MPM.addPass(RemoveDllImportPass());
                        return true;
                    }
                    return false;
                }
            );
        }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return getRemoveDllImportPassPluginInfo();
}
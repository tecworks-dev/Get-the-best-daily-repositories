#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <iostream>
#include <string>
#include <vector>

#include "../dllmapping.h"

using namespace llvm;

std::string recoverMicrosoftSymbol(const std::string &DecoratedName)
{
    if (DecoratedName[1] == '_')
    {  // for some reason [0] is empty?
        size_t atPos = DecoratedName.find('@');
        if (atPos != std::string::npos)
        {
            // Extract the portion between '_' and '@'
            return DecoratedName.substr(1 + 1, atPos - 1 - 1);
        }
    }
    return DecoratedName;
}

std::string GetWindowsDLLForFunction(const std::string &funcName)
{
    auto it = WindowsAPIMapping.find(funcName);
    return (it != WindowsAPIMapping.end()) ? it->second : "unknown.dll";
}

namespace
{
class DLLCallTransformPass : public PassInfoMixin<DLLCallTransformPass>
{
   public:
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM)
    {
        bool modified = false;

        // Declare the PSX_CALL function
        LLVMContext &context = M.getContext();
        Type        *intType = Type::getInt32Ty(context);
        Type        *stringType = Type::getInt8PtrTy(context);

        // PSX_CALL(char* dllName, char* funcName, int numArgs, ...)
        FunctionType *psxCallType = FunctionType::get(
            /* Return Type */ Type::getInt8PtrTy(context),
            /* Params */ {stringType, stringType, intType},
            /* IsVarArg */ true
        );
        FunctionCallee psxCallCallee = M.getOrInsertFunction("PSX_CALL", psxCallType);

        // for each function in module
        for (auto &F : M)
        {
            if (F.isDeclaration())
            {
                continue;
            }

            std::vector<CallInst *> callsToReplace;
            for (auto &BB : F)
            {
                for (auto &I : BB)
                {
                    if (auto *CI = dyn_cast<CallInst>(&I))
                    {
                        Function *calledFunc = CI->getCalledFunction();

                        // Check if this is an external function call
                        if (calledFunc && calledFunc->isDeclaration())
                        {
                            callsToReplace.push_back(CI);
                        }
                    }
                }
            }

            // Replace external function calls
            for (auto *CI : callsToReplace)
            {
                Function *originalFunc = CI->getCalledFunction();

                auto originalFuncName = recoverMicrosoftSymbol(originalFunc->getName().str());
                // Get the module name (DLL name)
                std::string dllName;
                if (!originalFunc->getName().empty())
                {
                    dllName = GetWindowsDLLForFunction(originalFuncName);
                }
                else
                {
                    continue;
                }

                if (originalFuncName == "PSX_PRINT")
                    continue;

                std::cout << "Replacing: " << dllName << " " << originalFuncName << std::endl;

                // Prepare builder to insert new call
                IRBuilder<> builder(CI);

                // Prepare arguments for PSX_CALL
                std::vector<Value *> psxCallArgs;
                Value               *dllNameVal = builder.CreateGlobalStringPtr(dllName);
                psxCallArgs.push_back(dllNameVal);
                Value *funcNameVal = builder.CreateGlobalStringPtr(originalFuncName);
                psxCallArgs.push_back(funcNameVal);
                Value *numArgsVal = ConstantInt::get(intType, CI->arg_size());
                psxCallArgs.push_back(numArgsVal);
                for (auto &arg : CI->args()) /* add remaining (original) arguments */
                    psxCallArgs.push_back(arg);

                // Create new PSX_CALL instruction
                CallInst *newCall =
                    builder.CreateCall(psxCallCallee.getFunctionType(), psxCallCallee.getCallee(), psxCallArgs);

                // Cast the return value of PSX_CALL if necessary
                Type  *expectedReturnType = CI->getType();
                Value *castedValue = newCall;

                if (expectedReturnType != newCall->getType())
                {
                    if (expectedReturnType->isPointerTy())
                    {
                        // cast to specific pointer type
                        castedValue = builder.CreateBitCast(newCall, expectedReturnType, "casted");
                    }
                    else if (expectedReturnType->isIntegerTy())
                    {
                        // cast to integer type
                        castedValue = builder.CreatePtrToInt(newCall, expectedReturnType, "casted");
                    }
                    else if (expectedReturnType->isFloatingPointTy())
                    {
                        errs() << "[!] Unsupported return type : " << *expectedReturnType << "\n";
                    }
                    else
                    {
                        errs() << "[!] Unexpected return type: " << *expectedReturnType << "\n";
                    }
                }

                // replace all uses of the original call and remove the original call instruction
                CI->replaceAllUsesWith(castedValue);
                CI->eraseFromParent();

                modified = true;
            }
        }

        return modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }

    static bool isRequired()
    {
        return true;
    }
};
}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return {
        LLVM_PLUGIN_API_VERSION, "DLLCallTransformPass", "v0.1",
        [](PassBuilder &PB)
        {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>)
                {
                    if (Name == "dll-call-transform")
                    {
                        MPM.addPass(DLLCallTransformPass());
                        return true;
                    }
                    return false;
                }
            );
        }};
}
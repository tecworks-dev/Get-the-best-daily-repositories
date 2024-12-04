#pragma once

#include <exception>

#include "Instruction.h"

#ifndef _DEBUG
#define _DEBUG
#include <stdio.h>
#endif
#include <windows.h>

#define MEMORY_SIZE        0xFFF
#define STACK_START        0x40000F00
#define TARGET_MEMORY_ADDR 0x40000000

#define NREGISTERS 32

typedef enum
{
    ZERO = 0,  // $0 - Always zero
    AT = 1,    // $1 - Assembler temporary
    V0 = 2,    // $2 - Function return value
    V1 = 3,    // $3 - Function return value
    A0 = 4,    // $4 - Function argument
    A1 = 5,    // $5 - Function argument
    A2 = 6,    // $6 - Function argument
    A3 = 7,    // $7 - Function argument
    T0 = 8,    // $8 - Temporary
    T1 = 9,    // $9 - Temporary
    T2 = 10,   // $10 - Temporary
    T3 = 11,   // $11 - Temporary
    T4 = 12,   // $12 - Temporary
    T5 = 13,   // $13 - Temporary
    T6 = 14,   // $14 - Temporary
    T7 = 15,   // $15 - Temporary
    S0 = 16,   // $16 - Saved register
    S1 = 17,   // $17 - Saved register
    S2 = 18,   // $18 - Saved register
    S3 = 19,   // $19 - Saved register
    S4 = 20,   // $20 - Saved register
    S5 = 21,   // $21 - Saved register
    S6 = 22,   // $22 - Saved register
    S7 = 23,   // $23 - Saved register
    T8 = 24,   // $24 - Temporary
    T9 = 25,   // $25 - Temporary
    K0 = 26,   // $26 - Reserved for OS kernel
    K1 = 27,   // $27 - Reserved for OS kernel
    GP = 28,   // $28 - Global pointer
    SP = 29,   // $29 - Stack pointer
    FP = 30,   // $30 - Frame pointer
    RA = 31,   // $31 - Return address
    HI = 32,   // $HI - Multiply/Divide high result
    LO = 33    // $LO - Multiply/Divide low result
} NamedRegister;

struct LoadRegister
{
    RegisterIndex registerIndex;
    uint32_t      value;
};

enum Exception
{
    SysCall = 0x8,           // caused by syscall opcode
    Overflow = 0xc,          // overflow on addi/add
    LoadAddressError = 0x4,  // if not 32 bit aligned
    StoreAddressError = 0x5,
    Break = 0x9,
    CoprocessorError = 0xb,
    IllegalInstruction = 0xa,
};

class PSX
{
   public:
    explicit PSX(const unsigned char *binary, const unsigned int &binarySize)
        : pc(TARGET_MEMORY_ADDR),
          current_pc(TARGET_MEMORY_ADDR),
          next_pc(TARGET_MEMORY_ADDR + 4),
          sr(0),
          hi(0),
          lo(0),
          load({{0}, 0})
    {
        // To pass arguments between the emulator and the host, we want to allocate emulated memory
        // at a specific address Specify a reasonable desired address, e.g., 0x40000000
        this->memory = (unsigned char *)VirtualAlloc(
            (void *)TARGET_MEMORY_ADDR, MEMORY_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE
        );
        if (this->memory == nullptr)
            return;

        // load binary to memory
        for (int i = 0; i < binarySize; ++i) this->memory[i] = binary[i];

        // set general purpose registers to default value
        for (uint32_t &reg : this->regs) reg = 0x00000000;

        // R0 is hardwired to 0
        this->regs[ZERO] = 0;

        // stack setup
        this->regs[SP] = STACK_START;
        this->regs[FP] = STACK_START;

        // out regs mirror the regs
        for (int i = 0; i < NREGISTERS; ++i) out_regs[i] = regs[i];

        isInitialized = true;
    }
    ~PSX()
    {
        VirtualFree(this->memory, 0, MEM_RELEASE);
    }

    bool RunNextInstruction();

    bool isInitialized = false;

   private:
    unsigned char *memory;
    bool           DEBUG = true;

    bool panic = false;

    // registers
    uint32_t pc;                     // Instruction Pointer (Program Counter)
    uint32_t current_pc;             // save address of current instruction to savein EPC in
                                     // case of exception
    uint32_t next_pc;                // simulate branch delay slot
    uint32_t regs[NREGISTERS] = {};  // 32 general purpose registers
    uint32_t sr;                     // cop0 register 12: status register
    uint32_t hi;                     // hi register for divison remainder and multiplication high result
    uint32_t lo;                     // lo register for division quotient and multiplication low result
    uint32_t cause;                  // cop0 register 13: cause register
    uint32_t epc;                    // cop0 register : exception PC

    // custom registers
    uint32_t out_regs[NREGISTERS] = {};  // second set of registers to emulate the load delay slot
                                         // accurately. contain output of the curren instruction
    LoadRegister load;                   // load initiated by the current instruction

    bool branching;    // if a branch occures, this is set to true
    bool inDelaySlot;  // if the current instruction is in the delay slot

    // Methods
    void Panic()
    {
        this->panic = true;
    }
    uint32_t GetRegister(const int &i) const;
    uint32_t GetRegister(const RegisterIndex &t) const;
    void     SetRegister(const RegisterIndex &t, const uint32_t &v);

    void Store8(const uint32_t &address, const uint8_t &value);
    void Store16(const uint32_t &address, const uint16_t &value);
    void Store32(const uint32_t &address, const uint32_t &value);

    uint8_t  Load8(const uint32_t &address) const;
    uint16_t Load16(const uint32_t &address) const;
    uint32_t Load32(const uint32_t &address) const;

    void Branch(const uint32_t &offset);
    void Interrupt(const Exception &interrupt);
    void DecodeAndExecute(const Instruction &instruction);

    void HostCall();

    // opcodes
    void OP_LUI(const Instruction &instruction);
    void OP_ORI(const Instruction &instruction);
    void OP_SW(const Instruction &instruction);
    void OP_LW(const Instruction &instruction);
    void OP_SLL(const Instruction &instruction);
    void OP_ADDIU(const Instruction &instruction);
    void OP_ADDI(const Instruction &instruction);
    void OP_ADDU(const Instruction &instruction);
    void OP_J(const Instruction &instruction);
    void OP_OR(const Instruction &instruction);
    void OP_BNE(const Instruction &instruction);
    void OP_SLTU(const Instruction &instruction);
    void OP_SH(const Instruction &instruction);
    void OP_JAL(const Instruction &instruction);
    void OP_ANDI(const Instruction &instruction);
    void OP_SB(const Instruction &instruction);
    void OP_JR(const Instruction &instruction);
    void OP_LB(const Instruction &instruction);
    void OP_BEQ(const Instruction &instruction);
    void OP_AND(const Instruction &instruction);
    void OP_JALR(const Instruction &instruction);
    void OP_MFC0(const Instruction &instruction);
    void OP_ADD(const Instruction &instruction);
    void OP_BGTZ(const Instruction &instruction);
    void OP_BLEZ(const Instruction &instruction);
    void OP_LBU(const Instruction &instruction);
    void OP_BXX(const Instruction &instruction);
    void OP_SLTI(const Instruction &instruction);
    void OP_SUBU(const Instruction &instruction);
    void OP_SRA(const Instruction &instruction);
    void OP_DIV(const Instruction &instruction);
    void OP_MFLO(const Instruction &instruction);
    void OP_SRL(const Instruction &instruction);
    void OP_SLTIU(const Instruction &instruction);
    void OP_DIVU(const Instruction &instruction);
    void OP_MFHI(const Instruction &instruction);
    void OP_SLT(const Instruction &instruction);
    void OP_SLLV(const Instruction &instruction);
    void OP_LH(const Instruction &instruction);
    void OP_XOR(const Instruction &instruction);
    void OP_SUB(const Instruction &instruction);
    void OP_MULT(const Instruction &instruction);
    void OP_BREAK(const Instruction &instruction);
    void OP_XORI(const Instruction &instruction);
    void OP_LWL(const Instruction &instruction);
    void OP_LWR(const Instruction &instruction);
    void OP_SWL(const Instruction &instruction);
    void OP_SWR(const Instruction &instruction);
    void OP_COP3(const Instruction &instruction);
    void OP_COP1(const Instruction &instruction);
    void OP_COP2(const Instruction &instruction);
    void OP_COP0(const Instruction &instruction);
    void OP_MTC0(const Instruction &instruction);
    void OP_LWC0(const Instruction &instruction);
    void OP_LWC1(const Instruction &instruction);
    void OP_LWC2(const Instruction &instruction);
    void OP_LWC3(const Instruction &instruction);
    void OP_SWC0(const Instruction &instruction);
    void OP_SWC1(const Instruction &instruction);
    void OP_SWC2(const Instruction &instruction);
    void OP_SWC3(const Instruction &instruction);
    void OP_SYSCALL(const Instruction &instruction);
    void OP_MTLO(const Instruction &instruction);
    void OP_MTHI(const Instruction &instruction);
    void OP_NOR(const Instruction &instruction);
    void OP_RFE(const Instruction &instruction);
    void OP_LHU(const Instruction &instruction);
    void OP_SRLV(const Instruction &instruction);
    void OP_ILLEGAL(const Instruction &instruction);
    void OP_MULTU(const Instruction &instruction);
    void OP_SRAV(const Instruction &instruction);

#ifdef _DEBUG
   public:
    void DbgDumpMemory(const int &addr, const int &length) const
    {
        for (int i = addr; i < addr + length; ++i)
        {
            printf(
                "0x%lx:\t%c (0x%lx)\n", i, this->memory[i - TARGET_MEMORY_ADDR], this->memory[i - TARGET_MEMORY_ADDR]
            );
        }
    }
    void DbgDisplayStack() const
    {
        auto stackpointer = this->regs[SP];

        for (int i = -16; i <= 32; i += 4)
        {
            if (i != 0)
                printf("     0x%lx (%i):\t0x%lx\n", stackpointer + i, i, this->Load32(stackpointer + i));
            else
                printf("sp-> 0x%lx (%i):\t0x%lx\n", stackpointer + i, i, this->Load32(stackpointer + i));
        }
    }
    void DbgPrintRegisters() const
    {
        printf("PC: 0x%lx\n", this->pc);
        for (int i = 0; i < NREGISTERS; i += 2)
        {
            printf("R%i:\t0x%lx\tR%i:\t0x%lx\n", i, this->regs[i], i + 1, this->regs[i + 1]);
        }
    }
#endif
};
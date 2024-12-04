#ifdef _DEBUG
#include <iostream>
#endif
#include "Psx.h"

// load upper immediate opcode:
// load value 'immediate' into upper 16 bits of target
void PSX::OP_LUI(const Instruction &instruction)
{
    auto immediate = instruction.imm();
    auto t = instruction.t();

    auto value = immediate << 16u;  // low 16bit set to 0

    this->SetRegister(t, value);
}

// bitwise or immediate opcode:
// bitwise or of value 'immediate' with source into target
void PSX::OP_ORI(const Instruction &instruction)
{
    auto immediate = instruction.imm();
    auto t = instruction.t();
    auto s = instruction.s();

    auto value = this->GetRegister(s) | immediate;

    this->SetRegister(t, value);
}

// store halfword
void PSX::OP_SH(const Instruction &instruction)
{
    if ((this->sr & 0x10000u) != 0u)
    {
        // cache is isolated, ignore writing
        printf("STUB:ignoring_store_while_cache_is_isolated", 0);
        return;
    }

    auto immediate = instruction.imm_se();  // SW is sign extending
    auto t = instruction.t();
    auto s = instruction.s();

    auto address = this->GetRegister(s) + immediate;

    // check alignment
    if (address % 2 != 0)
        return Interrupt(LoadAddressError);

    auto value = this->GetRegister(t);
    this->Store16(address, (uint16_t)value);
}

// store word opcode:
// store the word in target in source plus memory offset of immediate
void PSX::OP_SW(const Instruction &instruction)
{
    if ((this->sr & 0x10000u) != 0)
    {
        // cache is isolated, ignore writing
        printf("STUB:ignoring_store_while_cache_is_isolated");
        return;
    }

    auto immediate = instruction.imm_se();  // SW is sign extending
    auto t = instruction.t();
    auto s = instruction.s();

    auto address = this->GetRegister(s) + immediate;

    // check alignment
    if (address % 4 != 0)
        return Interrupt(LoadAddressError);

    auto value = this->GetRegister(t);

    this->Store32(address, value);
}

// load word opcode:
void PSX::OP_LW(const Instruction &instruction)
{
    if ((this->sr & 0x10000u) != 0)
    {
        // cache is isolated, ignore load
        printf("STUB:ignoring_load_while_cache_is_isolated", 0);
        return;
    }

    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto address = this->GetRegister(s) + immediate;

    // check alignment
    if (address % 4 != 0)
    {
        return Interrupt(LoadAddressError);
    }

    // TODO: workaround. debug _start epilogue
    if (address <= 0xFFFF)
    {
        printf("INVALID ADDR. 0x%lx\n", address);
        this->load = {t, 0};
        return;
    }

    auto value = this->Load32(address);

    // simulate loading delay by putting into laod registers
    this->load = {t, value};
}

// shift left logical
// shift bits from target by immediate to the left and store in destination
void PSX::OP_SLL(const Instruction &instruction)
{
    auto immediate = instruction.imm_shift();
    auto t = instruction.t();
    auto d = instruction.d();

    auto value = this->GetRegister(t) << immediate;
    this->SetRegister(d, value);
}

// add immediate unsigned
// name is misleading:
// we simply add immediate to source, save in target and truncate result on
// overflow
void PSX::OP_ADDIU(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto value = this->GetRegister(s) + immediate;
    this->SetRegister(t, value);
}

// check if theres an overflow happening on adding x and y
bool addOverflow(uint32_t x, uint32_t y, uint32_t &res)
{
    uint32_t temp = x + y;
    if (x > 0 && y > 0 && temp < 0)
        return true;
    if (x < 0 && y < 0 && temp > 0)
        return true;

    res = x + y;
    return false;
}

// add immediate
// we simply add immediate to source, save in target and throw an exception on
// overflow
void PSX::OP_ADDI(const Instruction &instruction)
{
    auto immediate = (int32_t)instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    uint32_t value;
    if (addOverflow(((int32_t)this->GetRegister(s)), immediate, value))
    {
        return this->Interrupt(Overflow);
    }
    this->SetRegister(t, value);
}

// add unsigned
// add two registers and store in d
void PSX::OP_ADDU(const Instruction &instruction)
{
    auto t = instruction.t();
    auto s = instruction.s();
    auto d = instruction.d();

    auto value = this->GetRegister(s) + this->GetRegister(t);
    this->SetRegister(d, value);
}

// jump
// set PC (instruction pointer) to address in immediate
void PSX::OP_J(const Instruction &instruction)
{
    auto immediate = instruction.imm_jump();
    this->branching = true;

    // immediate is shifted 2 to the right, because the two LSBs of pc are always
    // zero anyway (due to the 32bit boundary)
    this->next_pc = (this->next_pc & 0xf0000000u) | (immediate << 2u);
}

// jump and link
// jump and store return address in $ra ($31)
void PSX::OP_JAL(const Instruction &instruction)
{
    auto ra = this->next_pc;
    this->branching = true;

    // store return in ra
    this->SetRegister({31}, ra);

    this->OP_J(instruction);
}

// or
// bitwise or
void PSX::OP_OR(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();
    auto d = instruction.d();

    auto value = this->GetRegister(s) | this->GetRegister(t);
    this->SetRegister(d, value);
}

// and
// bitwise and
void PSX::OP_AND(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();
    auto d = instruction.d();

    auto value = this->GetRegister(s) & this->GetRegister(t);
    this->SetRegister(d, value);
}

// set on less than unsigned
// set rd to 0 or 1 depending on wheter rs is less than rt
void PSX::OP_SLTU(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();
    auto d = instruction.d();

    auto value = (uint32_t)(this->GetRegister(s) < this->GetRegister(t));
    this->SetRegister(d, value);
}

// branch to immediate value offset
void PSX::Branch(const uint32_t &offset)
{
    // offset immediates are shifted 2 to the right since PC/IP addresses are
    // aligned to 32bis
    auto off = offset << 2u;

    this->branching = true;

    this->next_pc = this->next_pc + off;
    this->next_pc = this->next_pc - 4;  // compensate for the pc += 4 of run_next_instruction
}

// branch (if) not equal
void PSX::OP_BNE(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto s = instruction.s();
    auto t = instruction.t();

    if (this->GetRegister(s) != this->GetRegister(t))
    {
        this->Branch(immediate);
    }
}

// bitwise and immediate
void PSX::OP_ANDI(const Instruction &instruction)
{
    auto immediate = instruction.imm();
    auto t = instruction.t();
    auto s = instruction.s();

    auto value = this->GetRegister(s) & immediate;

    this->SetRegister(t, value);
}

// coprocessor 0
void PSX::OP_COP0(const Instruction &instruction)
{
    switch (instruction.cop_opcode())
    {
        case 0b00100: this->OP_MTC0(instruction); break;
        case 0b00000: this->OP_MFC0(instruction); break;
        case 0b010000: this->OP_RFE(instruction); break;
        default:
            printf("Unhandled opcode 0x%lx", instruction.opcode);
            printf("Unhandled opcode for CoProcessor 0x%lx", instruction.cop_opcode());
            this->Panic();
    }
}

// move to coprocessor0 opcode
// loads bytes into a register of cop0
void PSX::OP_MTC0(const Instruction &instruction)
{
    auto cpu_r = instruction.t();
    auto cop_r = instruction.d().index;  // which register of cop0 to load into

    auto value = this->GetRegister(cpu_r);

    switch (cop_r)
    {
        case 3:
        case 5:
        case 6:
        case 7:
        case 9:
        case 11:  // breakpoint registers
            if (value != 0)
            {
                printf("Unhandled_write_to_cop0_register:_%lx", instruction.opcode);
                this->Panic();
            }
        case 12:  // status register
            this->sr = value;
            break;
        case 13:  // cause register, for exceptions
            if (value != 0)
            {
                printf("Unhandled_write_to_CAUSE_register:_%lx", instruction.opcode);
                this->Panic();
            }
        default: printf("STUB:Unhandled_cop0_register:_%lx", instruction.opcode);
    }
}

// store byte
void PSX::OP_SB(const Instruction &instruction)
{
    if ((this->sr & 0x10000u) != 0u)
    {
        // cache is isolated, ignore writing
        printf("STUB:ignoring_store_while_cache_is_isolated", 0);
        return;
    }

    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto address = this->GetRegister(s) + immediate;
    auto value = this->GetRegister(t);

    this->Store8(address, (uint8_t)value);
}

// jump register
// set PC to value stored in a register
void PSX::OP_JR(const Instruction &instruction)
{
    auto s = instruction.s();
    this->branching = true;

    auto t = this->GetRegister(s);

    // printf("JR %i: 0x%lx\n", s, t);

    this->next_pc = t;
}

// load byte (signed)
void PSX::OP_LB(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;

    // force sign extension by casting
    auto value = (int8_t)this->Load8(addr);

    // put load in the delay slot
    this->load = {t, (uint32_t)value};
}

// branch if equal
void PSX::OP_BEQ(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto s = instruction.s();
    auto t = instruction.t();

    if (this->GetRegister(s) == this->GetRegister(t))
    {
        this->Branch(immediate);
    }
}

// jump and link register
void PSX::OP_JALR(const Instruction &instruction)
{
    auto s = instruction.s();
    auto d = instruction.d();
    this->branching = true;

    this->SetRegister(d, this->next_pc);
    this->next_pc = this->GetRegister(s);
}

// move from coprocessor0
// read from a coprocessor register into target
void PSX::OP_MFC0(const Instruction &instruction)
{
    auto cpu_r = instruction.t();
    auto cop_r = instruction.d().index;

    uint32_t value;
    switch (cop_r)
    {
        case 12:  // status register
            value = this->sr;
            break;
        case 13:  // cause register, for exceptions
            value = this->cause;
            break;
        case 14:  // exception PC, store pc on esception
            value = this->epc;
            break;
        default: printf("STUB:Unhandled_read_from_cop0_register:_%lx", instruction.opcode); this->Panic();
    }

    this->load = {cpu_r, value};
}

// add , throw exception on signed overflow
void PSX::OP_ADD(const Instruction &instruction)
{
    auto t = instruction.t();
    auto s = instruction.s();
    auto d = instruction.d();

    auto s_add = (int32_t)this->GetRegister(s);
    auto t_add = (int32_t)this->GetRegister(t);

    uint32_t value;
    if (addOverflow(s_add, t_add, value))
        return this->Interrupt(Overflow);

    this->SetRegister(d, value);
}

// branch (if) greater than zero
void PSX::OP_BGTZ(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto s = instruction.s();

    auto val = (int32_t)this->GetRegister(s);  // sign cast necessary

    if (val > 0)
        this->Branch(immediate);
}

// branch (if) less (or) equal zero
void PSX::OP_BLEZ(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto s = instruction.s();

    auto val = (int32_t)this->GetRegister(s);  // sign cast necessary

    if (val <= 0)
        this->Branch(immediate);
}

// load byte unsigned
void PSX::OP_LBU(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;

    auto value = this->Load8(addr);

    // put load in the delay slot
    this->load = {t, (uint32_t)value};
}

// several opcodes: BLTZ, BLTZAL, BGEZ, BGEZAL
// bits 16 to 20 define which one
void PSX::OP_BXX(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto s = instruction.s();

    // if bit 16 is set, its BGEZ, otherwise BLTZ
    bool isBgez = (instruction.opcode >> 16u) & 1u;
    // if bits 20-17 are 0x80 then the return address is linked in $ra
    bool shouldLinkReturn = ((instruction.opcode >> 17u) & 0xfu) == 8;

    auto value = (int32_t)this->GetRegister(s);

    // test if LTZ
    auto test = (uint32_t)(value < 0);
    // if the test we want is GEZ, we negate the comparison above by XORing
    // this saves a branch and thus speeds it up
    test = test ^ isBgez;

    if (shouldLinkReturn)
    {
        auto ra = this->pc;
        this->SetRegister({31}, ra);
    }

    if (test != 0)
    {
        this->Branch(immediate);
    }
}

// substract unsigned
void PSX::OP_SUBU(const Instruction &instruction)
{
    auto t = instruction.t();
    auto s = instruction.s();
    auto d = instruction.d();

    auto value = this->GetRegister(s) - this->GetRegister(t);
    this->SetRegister(d, value);
}

// check if theres an overflow happening on substracting x and y
bool substractOverflow(uint32_t x, uint32_t y, uint32_t &res)
{
    uint32_t temp = x - y;
    if (x > 0 && y < 0 && temp < x)
        return true;
    if (x < 0 && y > 0 && temp > x)
        return true;

    res = x - y;
    return false;
}

// substract signed
void PSX::OP_SUB(const Instruction &instruction)
{
    auto t = instruction.t();
    auto s = instruction.s();
    auto d = instruction.d();

    uint32_t value;
    if (substractOverflow(((int32_t)this->GetRegister(s)), ((int32_t)this->GetRegister(t)), value))
    {
        return this->Interrupt(Overflow);
    }
    this->SetRegister(t, value);
}

// shift right arithmetic (arithmetic = signed)
void PSX::OP_SRA(const Instruction &instruction)
{
    auto immediate = instruction.imm_shift();
    auto t = instruction.t();
    auto d = instruction.d();

    // cast to signed to preserve sign bit
    auto value = ((int32_t)this->GetRegister(t)) >> immediate;

    this->SetRegister(d, (uint32_t)value);
}

// shift right logical (unsigned)
void PSX::OP_SRL(const Instruction &instruction)
{
    auto immediate = instruction.imm_shift();
    auto t = instruction.t();
    auto d = instruction.d();

    // cast to signed to preserve sign bit
    auto value = this->GetRegister(t) >> immediate;

    this->SetRegister(d, value);
}

// divide (signed)
void PSX::OP_DIV(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();

    auto n = (int32_t)this->GetRegister(s);  // numerator
    auto d = (int32_t)this->GetRegister(t);  // denominator -> n / d

    if (d == 0)
    {
        // division by zero, set bogus result
        this->hi = (uint32_t)n;
        if (n >= 0)
        {
            this->lo = 0xffffffff;
        }
        else
        {
            this->lo = 1;
        }
    }
    else if ((uint32_t)n == 0x80000000 && d == -1)
    {
        // result is not representable in 32 bit signed ints
        this->hi = 0;
        this->lo = 0x80000000;
    }
    else
    {
        this->hi = (uint32_t)(n % d);
        this->lo = (uint32_t)(n / d);
    }
}

// divide unsigned
void PSX::OP_DIVU(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();

    auto n = this->GetRegister(s);  // numerator
    auto d = this->GetRegister(t);  // denominator -> n / d

    if (d == 0)
    {
        // division by zero, set bogus result
        this->hi = n;
        this->lo = 0xffffffff;
    }
    else
    {
        this->hi = n % d;
        this->lo = n / d;
    }
}

// move from lo-register
void PSX::OP_MFLO(const Instruction &instruction)
{
    // TODO should stall if division not done yet
    auto d = instruction.d();
    this->SetRegister(d, this->lo);
}

// set less than immediate unsigned
void PSX::OP_SLTIU(const Instruction &instruction)
{
    auto immediate = (int32_t)instruction.imm_se();
    auto s = instruction.s();
    auto t = instruction.t();

    auto value = this->GetRegister(s) < immediate;
    this->SetRegister(t, (uint32_t)value);
}

// set less than immediate
// set t to 1 if s less than immediate else to 0
void PSX::OP_SLTI(const Instruction &instruction)
{
    auto immediate = (int32_t)instruction.imm_se();
    auto s = instruction.s();
    auto t = instruction.t();

    auto value = ((int32_t)this->GetRegister(s)) < immediate;
    this->SetRegister(t, (uint32_t)value);
}

// move from hi
void PSX::OP_MFHI(const Instruction &instruction)
{
    // TODO should stall if division not done yet
    auto d = instruction.d();
    this->SetRegister(d, this->hi);
}

// set on less than (signed)
void PSX::OP_SLT(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();
    auto d = instruction.d();

    auto value = ((int32_t)this->GetRegister(s)) < ((int32_t)this->GetRegister(t));
    this->SetRegister(d, (uint32_t)value);
}

void PSX::OP_SYSCALL(const Instruction &instruction)
{
    this->Interrupt(SysCall);
}

// move to LO
void PSX::OP_MTLO(const Instruction &instruction)
{
    auto s = instruction.s();
    this->lo = this->GetRegister(s);
}

// move to HI
void PSX::OP_MTHI(const Instruction &instruction)
{
    auto s = instruction.s();
    this->hi = this->GetRegister(s);
}

void PSX::OP_NOR(const Instruction &instruction)
{
    auto d = instruction.d();
    auto s = instruction.s();
    auto t = instruction.t();
    auto value = ~(this->GetRegister(s) | this->GetRegister(t));
    this->SetRegister(d, value);
}

// return from exceptions
void PSX::OP_RFE(const Instruction &instruction)
{
    // there are more instructions with the same encoding, which the playstation
    // does not use since they are virtual memory related. still check for buggy
    // code
    if ((instruction.opcode & 0x3fu) != 0b010000)
    {
        printf("Invalid_cop0_instruction:_%lx", instruction.opcode);
    }

    // restore the pre-exception mode by shifting the interrupt bits of the status
    // register back
    auto mode = this->sr & 0x3fu;
    this->sr &= ~0x3fu;
    this->sr |= mode >> 2u;
}

// load halfword unsigned
void PSX::OP_LHU(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;

    if (addr % 2 != 0)
    {
        return this->Interrupt(LoadAddressError);
    }

    auto value = this->Load16(addr);
    // put load in the delay slot
    this->load = {t, (uint32_t)value};
}

// shift left logical variable
void PSX::OP_SLLV(const Instruction &instruction)
{
    auto t = instruction.t();
    auto s = instruction.s();
    auto d = instruction.d();

    // shift amount is truncated to lower 5 bits
    auto value = this->GetRegister(t) << (this->GetRegister(s) & 0x1fu);

    this->SetRegister(d, value);
}

// load halfword
void PSX::OP_LH(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;

    // force sign extension by casting
    auto value = (int16_t)this->Load16(addr);
    // put load in the delay slot
    this->load = {t, (uint32_t)value};
}

// exclusive or
void PSX::OP_XOR(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();
    auto d = instruction.d();

    auto value = this->GetRegister(s) ^ this->GetRegister(t);
    this->SetRegister(d, value);
}

// exclusive or immediate
void PSX::OP_XORI(const Instruction &instruction)
{
    auto immediate = instruction.imm();
    auto t = instruction.t();
    auto s = instruction.s();

    auto value = this->GetRegister(s) ^ immediate;
    this->SetRegister(t, value);
}

// break
void PSX::OP_BREAK(const Instruction &instruction)
{
    this->Interrupt(Break);
}

// multiply (signed)
void PSX::OP_MULT(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();

    // sign extend by casting
    auto a = (int64_t)((int32_t)this->GetRegister(s));
    auto b = (int64_t)((int32_t)this->GetRegister(t));

    auto value = (uint64_t)(a * b);

    this->hi = (uint32_t)(value >> 32u);
    this->lo = (uint32_t)value;
}

// multiply (unsigned)
void PSX::OP_MULTU(const Instruction &instruction)
{
    auto s = instruction.s();
    auto t = instruction.t();

    auto a = (uint64_t)this->GetRegister(s);
    auto b = (uint64_t)this->GetRegister(t);

    auto value = a * b;

    this->hi = (uint32_t)(value >> 32u);
    this->lo = (uint32_t)value;
}

// coprocessor 1 opcode does not exist on the playstation
void PSX::OP_COP1(const Instruction &instruction)
{
    this->Interrupt(CoprocessorError);
}

// coprocessor 3 opcode does not exist on the playstation
void PSX::OP_COP3(const Instruction &instruction)
{
    this->Interrupt(CoprocessorError);
}

// coprocessor 2, GTE (geometry transform engine)
void PSX::OP_COP2(const Instruction &instruction)
{
    printf("STUB:unhandled_GTE_instruction:_x0%lx", instruction.opcode);
    this->Panic();
}

// load word left (little endian only)
void PSX::OP_LWL(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;

    // this instruction bypasses load delay restrictions:
    auto curValue = this->out_regs[t.index];

    // next, load the aligned word containing the first addressed byte
    auto alignedAddr = addr & ~3u;
    auto alignedWord = this->Load32(alignedAddr);

    // depneding on the address alignment, we fetch the 1-4 most significant bytes
    // and put them in the target register
    uint32_t value;
    switch (addr & 3u)
    {
        case 0: value = (curValue & 0x00ffffffu) | (alignedWord << 24u); break;
        case 1: value = (curValue & 0x0000ffffu) | (alignedWord << 16u); break;
        case 2: value = (curValue & 0x000000ffu) | (alignedWord << 8u); break;
        case 3: value = (curValue & 0x00000000u) | (alignedWord << 0u); break;
    }

    this->load = {t, value};
}

// load word right (little endian only)
void PSX::OP_LWR(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;

    // this instruction bypasses load delay restrictions:
    auto curValue = this->out_regs[t.index];

    // next, load the aligned word containing the first addressed byte
    auto alignedAddr = addr & ~3u;
    auto alignedWord = this->Load32(alignedAddr);

    // depneding on the address alignment, we fetch the 1-4 most significant bytes
    // and put them in the target register
    uint32_t value;
    switch (addr & 3u)
    {
        case 1: value = (curValue & 0x00000000u) | (alignedWord >> 0u); break;
        case 2: value = (curValue & 0xff000000u) | (alignedWord >> 8u); break;
        case 0: value = (curValue & 0xffff0000u) | (alignedWord >> 16u); break;
        case 3: value = (curValue & 0xffffff00u) | (alignedWord >> 24u); break;
    }

    this->load = {t, value};
}

// store word left
void PSX::OP_SWL(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;
    auto value = this->GetRegister(t);

    auto alignedAddr = addr & ~3u;
    auto curMem = this->Load32(alignedAddr);

    uint32_t mem;
    switch (addr & 3u)
    {
        case 0: mem = (curMem & 0xffffff00u) | (value >> 24u); break;
        case 1: mem = (curMem & 0xffff0000u) | (value >> 16u); break;
        case 2: mem = (curMem & 0xff000000u) | (value >> 8u); break;
        case 3: mem = (curMem & 0x00000000u) | (value >> 0u); break;
    }

    this->Store32(alignedAddr, mem);
}

// store word right
void PSX::OP_SWR(const Instruction &instruction)
{
    auto immediate = instruction.imm_se();
    auto t = instruction.t();
    auto s = instruction.s();

    auto addr = this->GetRegister(s) + immediate;
    auto value = this->GetRegister(t);

    auto alignedAddr = addr & ~3u;
    auto curMem = this->Load32(alignedAddr);

    uint32_t mem;
    switch (addr & 3u)
    {
        case 0: mem = (curMem & 0x00000000u) | (value << 0u); break;
        case 1: mem = (curMem & 0x000000ffu) | (value << 8u); break;
        case 2: mem = (curMem & 0x0000ffffu) | (value << 16u); break;
        case 3: mem = (curMem & 0x00ffffffu) | (value << 24u); break;
    }

    this->Store32(alignedAddr, mem);
}

// load word coprocessor n
void PSX::OP_LWC0(const Instruction &instruction)
{
    // not supported by c0
    this->Interrupt(CoprocessorError);
}
void PSX::OP_LWC1(const Instruction &instruction)
{
    // not supported by c1
    this->Interrupt(CoprocessorError);
}
void PSX::OP_LWC2(const Instruction &instruction)
{
    printf("Unhandled_GTE_LWC_instruction:_0x%lx", instruction.opcode);
    this->Panic();
}
void PSX::OP_LWC3(const Instruction &instruction)
{
    // not supported by c3
    this->Interrupt(CoprocessorError);
}

// store word coprocessor n
void PSX::OP_SWC0(const Instruction &instruction)
{
    // not supported by c0
    this->Interrupt(CoprocessorError);
}
void PSX::OP_SWC1(const Instruction &instruction)
{
    // not supported by c1
    this->Interrupt(CoprocessorError);
}
void PSX::OP_SWC2(const Instruction &instruction)
{
    printf("Unhandled_GTE_SWC_instruction:_0x%lx", instruction.opcode);
}
void PSX::OP_SWC3(const Instruction &instruction)
{
    // not supported by c3
    this->Interrupt(CoprocessorError);
}

void PSX::OP_ILLEGAL(const Instruction &instruction)
{
    printf("Illegal_instruction:_0x%lx", instruction.opcode);
    this->Interrupt(IllegalInstruction);
}

// shift right logical variable
void PSX::OP_SRLV(const Instruction &instruction)
{
    auto t = instruction.t();
    auto s = instruction.s();
    auto d = instruction.d();

    // shift amount truncated to 5 bits
    auto value = this->GetRegister(t) >> (this->GetRegister(s) & 0x1fu);

    this->SetRegister(d, value);
}

// shift right arithmetic variable
void PSX::OP_SRAV(const Instruction &instruction)
{
    auto t = instruction.t();
    auto s = instruction.s();
    auto d = instruction.d();

    // shift amount truncated to 5 bits
    auto value = ((int32_t)this->GetRegister(t)) >> (this->GetRegister(s) & 0x1fu);

    this->SetRegister(d, (uint32_t)value);
}
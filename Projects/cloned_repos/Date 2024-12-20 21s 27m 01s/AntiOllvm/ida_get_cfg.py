import idaapi
import idautils
import idc
import json


def get_func_flowchart(func_addr):
    """
    获取指定函数的控制流图，返回包含基本块信息的 FlowChart 对象
    """
    func = idaapi.get_func(func_addr)
    if not func:
        print(f"无法获取函数 {hex(func_addr)}")
        return []
    flowchart = idaapi.FlowChart(func)
    return flowchart


def get_basic_block_instructions(block_start, block_end):
    """
    获取基本块内的所有汇编指令和它们的偏移
    """
    instructions = []
    ea = block_start
    while ea < block_end:
        mnem = idc.print_insn_mnem(ea)

        # 解码指令以获取操作数
        insn = idaapi.insn_t()
        size = idaapi.decode_insn(insn, ea)
        if size == 0:
            print(f"无法解码指令: {hex(ea)}")
            break  # 退出循环以避免无限循环

        machine_code_bytes = idc.get_bytes(ea, size)
        if machine_code_bytes:
            machine_code = machine_code_bytes.hex()
        else:
            machine_code = ""

        operands = []
        for i in range(idaapi.UA_MAXOP):
            if insn.ops[i].type == idaapi.o_void:
                break
            operand = idc.print_operand(ea, i)
            operands.append(operand)
            # json序列化ops[i]

            # print()
        operands_str = ','.join(operands)

        instructions.append({
            "address": hex(ea),
            "mnemonic": mnem,
            "operands_str": operands_str,
            "machine_code": machine_code
        })

        # 获取下一条指令的地址
        ea = idc.next_head(ea, block_end)
    return instructions


def export_cfg(func_addr):
    """
    导出指定函数的控制流图，返回基本块及跳转关系
    """
    flowchart = get_func_flowchart(func_addr)
    cfg = []

    for block in flowchart:
        # 获取每个基本块的起始地址和结束地址
        block_start = block.start_ea
        block_end = block.end_ea

        # 获取基本块内的汇编指令及偏移
        instructions = get_basic_block_instructions(block_start, block_end)

        # 获取该基本块的后继基本块
        successors = []
        for succ in block.succs():
            successors.append(hex(succ.start_ea))

        cfg.append({
            "start_address": hex(block_start),
            "end_address": hex(block_end),
            "instructions": instructions,
            "linkBlocks": successors
        })

    return cfg


def main():
    # choose your function address
    func_addr = 0x181c6c  # replace with your function address

    # 确保函数地址有效
    func = idaapi.get_func(func_addr)
    if not func:
        print(f"地址 {hex(func_addr)} 不是一个有效的函数入口。")
        return

    # 导出该函数的控制流图信息
    cfg = export_cfg(func_addr)
    # 获取函数结束地址
    func_end = func.end_ea

    result_data = {
        "function_start": hex(func_addr),
        "function_end": hex(func_end),
        "cfg": cfg
    }
   
    output_file = "C:/Users/PC5000/PycharmProjects/py_ida/cfg_output_" + hex(func_addr) + ".json"
    try:
        with open(output_file, 'w') as json_file:
            json.dump(result_data, json_file, indent=4)
        print(f"CFG is save to {output_file}")
    except Exception as e:
        print(f"save JSON file Error: {e}")


if __name__ == "__main__":
    main()

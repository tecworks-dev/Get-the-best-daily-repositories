import json
from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN, KsError

# JSON 文件路径
json_file_path = "fix.json"

# 读取 JSON 数据
try:
    with open(json_file_path, "r") as f:
        patches = json.load(f)
except Exception as e:
    print(f"读取 JSON 文件时出错: {e}")
    patches = []

# 初始化 Keystone 引擎（ARM64，小端模式）
try:
    ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
except KsError as e:
    print(f"Keystone 初始化错误: {e}")
    ks = None

if not ks:
    print("Keystone 引擎未能正确初始化。")
else:
    # 遍历每一个补丁项
    for item in patches:
        addr_str = item.get("address")
        fix_code_str = item.get("fixmachine_code")

        if not addr_str or not fix_code_str:
            fixmachine_byteCode = item.get("fixmachine_byteCode")
            if not fixmachine_byteCode:
                print(f"跳过无效项: {item}")
                continue
            else:
                item["fix_machine_code_bytes"] = fixmachine_byteCode
            continue

        try:
            # 将地址字符串转换为整数
            addr = int(addr_str, 16)
        except ValueError:
            print(f"无效的地址格式: {addr_str}")
            continue

        # 汇编 fixmachine_code
        try:
            encoding, count = ks.asm(fix_code_str, addr)
            machine_code_hex = ''.join(f"{byte:02x}" for byte in encoding)
            print(f"指令 '{fix_code_str}' 汇编为: {machine_code_hex}")

            # 将机器码十六进制字符串添加到补丁项
            item["fix_machine_code_bytes"] = machine_code_hex
        except KsError as e:
            #打印下地址
            print(f"汇编指令 '{fix_code_str}' 时出错: {e} at {addr_str}")

            item["fix_machine_code_bytes"] = None

    # 将更新后的补丁项写回 JSON 文件
    try:
        with open(json_file_path, "w") as f:
            json.dump(patches, f, indent=4)
        print("更新后的 JSON 文件已保存。")
    except Exception as e:
        print(f"保存更新后的 JSON 文件时出错: {e}")

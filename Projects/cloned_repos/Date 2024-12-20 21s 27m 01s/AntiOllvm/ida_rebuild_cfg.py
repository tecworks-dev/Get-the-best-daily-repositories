import json
import idc
import ida_bytes
import idaapi
import ida_kernwin
# 选择 JSON 文件
json_file_path = ida_kernwin.ask_file(0, "*.json", "please choose fix.json when gen_machine_code.py is executed")

if not json_file_path:
    print("未选择文件，脚本退出。")
else:
    # 读取并解析 JSON 文件
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)


    # 遍历每一个补丁项
    for item in data:
        addr_str = item.get("address")
        fix_code_bytes_str = item.get("fix_machine_code_bytes")

        if not addr_str or not fix_code_bytes_str:
            print(f"跳过无效项: {item}")
            continue

        try:
            # 将地址字符串转换为整数
            addr = int(addr_str, 16)
        except ValueError:
            print(f"无效的地址格式: {addr_str}")
            continue

        try:
            # 将十六进制字符串转换为字节
            machine_bytes = bytes.fromhex(fix_code_bytes_str)
            print(f"机器码 '{fix_code_bytes_str}' 转换为字节。")
        except ValueError as e:
            print(f"转换机器码 '{fix_code_bytes_str}' 时出错: {e}")
            continue

        # 检查地址是否在可写范围内
        if not ida_bytes.is_mapped(addr):
            print(f"地址 {addr_str} 未在当前二进制文件中映射。")
            continue

        try:
            # 补丁机器码到指定地址
            ida_bytes.patch_bytes(addr, machine_bytes)
            print(f"成功补丁地址 {addr_str} 以机器码 '{fix_code_bytes_str}'.")
        except Exception as e:
            print(f"在地址 {addr_str} 补丁时出错: {e}")
            continue

    # 刷新 IDA 的显示
    idaapi.refresh_idaview_anyway()
    print("补丁操作完成。")

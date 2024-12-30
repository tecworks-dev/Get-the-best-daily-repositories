# WSTG-ERRH-01: Testing for Improper Error Handling

## Summary
Improper error handling occurs when an application exposes sensitive information through error messages. This can include stack traces, database errors, or system information that could aid an attacker in identifying vulnerabilities.

---

## Objectives
- Identify endpoints that disclose sensitive information via error messages.
- Test for improper error handling and analyze the type of information exposed.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Locate endpoints that process user inputs, such as forms, API requests, or file uploads.
   - Example:
     ```http
     GET /search?query=test HTTP/1.1
     Host: example.com
```     

2. **Trigger Errors:**
   - Use invalid inputs or manipulate requests to trigger errors:
     - **SQL errors:**
       ```sql
       ' OR 1=1 --
```       
     - **File errors:**
       ```plaintext
       ../../../etc/passwd
       ```
     - **API errors:**
       ```json
       {
         "invalid": "json"
       }
       ```

3. **Analyze Error Messages:**
   - Check for:
     - Detailed stack traces.
     - SQL error messages revealing query structure.
     - Information about server, framework, or database versions.

4. **Test for Authentication or Authorization Errors:**
   - Attempt unauthorized actions and analyze error responses.
   - Example:
     ```http
     GET /admin HTTP/1.1
     Host: example.com
```     

---

### Automated Testing

Use tools like:
- **Burp Suite:** Analyze responses for error messages.
- **OWASP ZAP:** Scan for improper error handling issues.
- **Nikto:** Identify exposed error messages in HTTP responses.

#### Example with Burp Suite:
- Intercept and modify requests to inject invalid inputs.
- Analyze the responses for error disclosures.

---

## Mitigation
- Implement generic error messages that do not reveal sensitive information.
- Log detailed error information on the server-side for debugging purposes.
- Use a centralized error handling mechanism to enforce consistent error messages.
- Regularly test and review error handling implementations.

---

## References
- OWASP Improper Error Handling Cheat Sheet
- CWE-209: Information Exposure Through an Error Message
- Tools:
  - [Burp Suite](https://portswigger.net/burp)
  - [OWASP ZAP](https://www.zaproxy.org/)
  - [Nikto](https://cirt.net/Nikto2)
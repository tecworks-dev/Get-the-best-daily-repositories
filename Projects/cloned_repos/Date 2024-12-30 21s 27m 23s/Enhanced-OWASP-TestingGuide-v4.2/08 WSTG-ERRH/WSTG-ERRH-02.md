# WSTG-ERRH-02: Testing for Stack Traces and Error Messages Disclosure

## Summary
Stack traces and detailed error messages may expose sensitive application internals, including class names, methods, database structures, and server information. This information can aid attackers in crafting targeted attacks.

---

## Objectives
- Detect if stack traces or verbose error messages are disclosed.
- Assess the type and sensitivity of information exposed.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Locate application areas where user input is processed.
   - Example request:
     ```http
     GET /search?query=test HTTP/1.1
     Host: example.com
```     

2. **Trigger Stack Traces or Errors:**
   - Use invalid inputs or malformed requests to induce application errors:
     - **SQL errors:**
       ```sql
       ' OR 1=1 --
```       
     - **File errors:**
       ```plaintext
       ../../../nonexistent_file
       ```
     - **API errors:**
       ```json
       {
         "invalid": "data"
       }
       ```

3. **Analyze Error Messages:**
   - Look for detailed information in the response, such as:
     - File paths.
     - Class names or methods.
     - Database schemas or queries.
     - Server software and version details.
   - Example response:
     ```plaintext
     java.lang.NullPointerException
     at com.example.project.module.Class.method(Class.java:123)
     ```

4. **Test for Custom Error Pages:**
   - Verify if custom error pages are implemented for common errors (e.g., 404, 500).
   - Example malformed request:
     ```http
     GET /this_page_does_not_exist HTTP/1.1
     Host: example.com
```     

---

### Automated Testing

Use tools like:
- **Burp Suite:** Analyze responses for stack traces and verbose error messages.
- **OWASP ZAP:** Scan for error disclosures in responses.
- **Nikto:** Detect exposed server and application error messages.

#### Example with OWASP ZAP:
- Run the active scanner and check for detailed error disclosures in HTTP responses.

---

## Mitigation
- Implement custom error pages that return generic error messages.
- Disable detailed stack traces and error messages in production environments.
- Log detailed errors server-side and ensure they are not exposed to the client.
- Conduct regular application security reviews to detect and fix error-handling issues.

---

## References
- OWASP Error Handling Cheat Sheet
- CWE-209: Information Exposure Through an Error Message
- Tools:
  - [Burp Suite](https://portswigger.net/burp)
  - [OWASP ZAP](https://www.zaproxy.org/)
  - [Nikto](https://cirt.net/Nikto2)

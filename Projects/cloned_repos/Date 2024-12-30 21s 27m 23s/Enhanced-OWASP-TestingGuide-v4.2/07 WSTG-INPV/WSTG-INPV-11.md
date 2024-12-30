# WSTG-INPV-11: Testing for NoSQL Injection

## Summary
NoSQL Injection occurs when an attacker manipulates NoSQL queries to bypass authentication, retrieve unauthorized data, or compromise the underlying NoSQL database. Since NoSQL databases often use JSON-like queries, injection can occur via unsanitized user inputs.

---

## Objectives
- Identify application endpoints that interact with NoSQL databases.
- Test for injection vulnerabilities in these endpoints.
- Assess the potential impact of exploiting NoSQL injection.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Map all user inputs that interact with the NoSQL database.
   - Focus on JSON-based queries, URL parameters, headers, and cookies.

2. **Inject Malicious Payloads:**
   - Test for injection vulnerabilities by crafting malicious NoSQL queries:
     ```json
     { "username": { "$ne": null }, "password": { "$ne": null } }
     ```
   - Example:

 ```http
     POST /login HTTP/1.1
     Content-Type: application/json

     {
       "username": { "$ne": null },
       "password": { "$ne": null }
     }
```     

3. **Observe Application Behavior:**
   - Look for signs of authentication bypass or unexpected results.
   - Check for database errors or unintended responses.

4. **Blind NoSQL Injection Testing:**
   - Use conditional queries to infer database behavior:
     ```json
     { "username": "admin", "password": { "$gt": "" } }
     ```
   - Observe differences in responses or response times.

5. **Advanced Testing:**
   - Test for extraction of sensitive data using crafted queries:
     ```json
     { "username": { "$regex": "^.*$" }, "password": { "$regex": "^.*$" } }
     ```

---

### Automated Testing

Use tools like:
- **NoSQLMap:** Automate NoSQL injection detection and exploitation.
- **Burp Suite Extensions:** Plugins for testing NoSQL vulnerabilities.

#### Example with NoSQLMap:
- Run NoSQLMap against a vulnerable endpoint:
  ```bash
  nosqlmap -u "http://example.com/login" -d '{"username":"admin","password":"123"}'
  ```
- Analyze the results to identify potential vulnerabilities.

---

## Mitigation
- Validate and sanitize all user inputs to ensure they conform to expected formats.
- Use parameterized queries or query builders that safely handle user input.
- Implement robust authentication and authorization mechanisms.
- Monitor database queries for unusual patterns or malicious behavior.
- Regularly update and patch NoSQL database software.

---

## References
- OWASP NoSQL Injection Cheat Sheet
- CWE-943: Improper Neutralization of Special Elements in Data Query Logic
- Tools:
  - [NoSQLMap](https://github.com/codingo/NoSQLMap)
  - [Burp Suite Extensions](https://portswigger.net/bappstore)

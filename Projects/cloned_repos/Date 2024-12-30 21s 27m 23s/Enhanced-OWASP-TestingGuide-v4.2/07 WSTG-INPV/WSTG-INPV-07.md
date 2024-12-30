# WSTG-INPV-07: Testing for LDAP Injection

## Summary
LDAP Injection occurs when user input is improperly sanitized and incorporated into an LDAP query. Attackers can exploit this vulnerability to manipulate LDAP queries, potentially bypassing authentication, accessing unauthorized information, or escalating privileges.

## Objective
To identify and exploit LDAP Injection vulnerabilities in the target application.

## How to Test

### 1. Manual Testing
#### a. Input Validation Testing
- Inject LDAP-specific payloads into input fields to observe application behavior:
  - `*)(&`
  - `(uid=*))(|(uid=*))`
  - `(&(objectClass=*))`
- Look for error messages or unexpected application responses.

#### b. URL Parameter Testing
- Test query parameters in the URL with LDAP payloads:
  - `?user=*)(uid=*))`
  - `?search=(&(objectClass=*))`
- Analyze application behavior for anomalies.

#### c. Header and Cookie Testing
- Insert LDAP payloads into HTTP headers like `User-Agent`, `Referer`, or `Cookie`.

### 2. Blind LDAP Injection
If no direct errors are visible:
- Use boolean-based techniques:
  - `(uid=admin)(!(uid=*))`
- Time-based techniques (if supported by the application).

### 3. Exploit LDAP Queries
- Enumerate objects by modifying query filters:
  - `(objectClass=*)`
  - `(uid=*))(|(uid=*))(&(objectClass=*))`
- Bypass authentication:
  - `admin*)(|(password=*))`

### 4. Automated Testing
Use tools to automate LDAP Injection detection:
- **Burp Suite**: Intercept and manipulate requests with LDAP payloads.
- **OWASP ZAP**: Perform injection tests automatically.

## Tools
- **LDAPSearch**: Query and test LDAP directories.
- **Burp Suite**: Modify and test LDAP-related requests.
- **OWASP ZAP**: Automated injection testing.

## Remediation
1. **Input Validation**: Implement strict allowlists to validate user input.
2. **Parameterized Queries**: Use prepared LDAP queries to prevent injection.
3. **Error Handling**: Suppress detailed LDAP error messages.
4. **Access Controls**: Restrict LDAP query privileges to minimize impact.

## References
- [OWASP LDAP Injection](https://owasp.org/www-community/attacks/LDAP_Injection)
- [LDAP Search Filter Syntax](https://tools.ietf.org/html/rfc4515)
- [NIST Directory Services Guidelines](https://csrc.nist.gov/publications)

# WSTG-INPV-08: Testing for XPath Injection

## Summary
XPath Injection occurs when user input is improperly sanitized and directly incorporated into XPath queries. Attackers can exploit this vulnerability to manipulate XML data, retrieve unauthorized information, or bypass authentication mechanisms.

## Objective
To identify and exploit XPath Injection vulnerabilities in the target application.

## How to Test

### 1. Manual Testing
#### a. Input Validation Testing
- Inject XPath-specific payloads into input fields to observe application behavior:
  - `' or '1'='1`
  - `" or "1"="1`
  - `') or ('1'='1`
  - `admin' and '1'='1`
- Look for unexpected results, such as authentication bypass or data leakage.

#### b. URL Parameter Testing
- Test query parameters in the URL with XPath payloads:
  - `?search=') or ('1'='1`
  - `?id=' or '1'='1`
- Analyze application behavior for anomalies.

#### c. Header and Cookie Testing
- Insert XPath payloads into HTTP headers like `User-Agent`, `Referer`, or `Cookie`.

### 2. Blind XPath Injection
If no direct errors are visible:
- Use boolean-based techniques:
  - `username='admin' and 1=1 or 'a'='b`
  - `username='admin' and 1=2 or 'a'='a`
- Time-based techniques (if supported):
  - Inject payloads that exploit processing delays.

### 3. Exploiting XPath Queries
- Enumerate nodes by modifying queries:
  - `/users/user[username/text()='admin']`
  - `/users/user[position()=1]`
- Bypass authentication:
  - `username='admin' or '1'='1`

### 4. Automated Testing
Use tools to automate XPath Injection detection:
- **Burp Suite**: Intercept and manipulate requests with XPath payloads.
- **OWASP ZAP**: Perform injection tests automatically.

## Tools
- **XPath Tester**: Validate XPath queries.
- **Burp Suite**: Modify and test XPath-related requests.
- **OWASP ZAP**: Automated injection testing.

## Remediation
1. **Input Validation**: Implement strict allowlists to validate user input.
2. **Parameterized Queries**: Use prepared XPath queries to prevent injection.
3. **Error Handling**: Suppress detailed XPath error messages.
4. **Access Controls**: Restrict access to sensitive XML data.

## References
- [OWASP XPath Injection](https://owasp.org/www-community/attacks/XPATH_Injection)
- [W3C XPath Specifications](https://www.w3.org/TR/xpath/)
- [NIST XML Security Guidelines](https://csrc.nist.gov/publications)

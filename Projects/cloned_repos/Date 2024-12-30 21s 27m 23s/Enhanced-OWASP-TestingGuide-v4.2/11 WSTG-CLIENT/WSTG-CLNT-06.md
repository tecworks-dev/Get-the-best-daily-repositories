Here’s the documentation for **WSTG-CLNT-06** in Markdown format:


# WSTG-CLNT-06: Testing for Client-Side Storage

## Summary

Modern web applications often store data on the client-side using mechanisms such as cookies, localStorage, sessionStorage, IndexedDB, or Web SQL. Improper use of client-side storage can lead to vulnerabilities such as unauthorized data access, data leakage, or manipulation by attackers.

## Objective

To identify and exploit weaknesses in the application’s client-side storage mechanisms, including:

- Sensitive data stored insecurely on the client.
- Lack of encryption or obfuscation.
- Weak validation or reliance on client-side data.

## How to Test

### Step 1: Identify Storage Mechanisms
1. Use browser developer tools to inspect the following storage mechanisms:
   - **Cookies**:
     - Check for sensitive data stored in cookies (e.g., tokens, session IDs).
   - **localStorage and sessionStorage**:
     - Identify keys and values stored.
   - **IndexedDB and Web SQL**:
     - Inspect databases and their contents.

2. Determine if sensitive data (e.g., passwords, tokens, user information) is stored in any of these mechanisms.

---

### Step 2: Analyze Data Stored on the Client
1. Look for insecure storage practices, such as:
   - Storing sensitive data (e.g., passwords, tokens) without encryption.
   - Using client-side storage for critical application logic.

2. Identify mechanisms used to secure stored data:
   - Encryption or obfuscation techniques.
   - Validation mechanisms on the server-side.

---

### Step 3: Perform Security Tests
1. **Tamper with Stored Data**:
   - Modify values in localStorage, sessionStorage, or cookies to test for vulnerabilities.
   - Example: Change a `role` value from `user` to `admin`.

2. **Access Data Across Sessions**:
   - Test if session-specific data is improperly stored in persistent storage (e.g., localStorage).

3. **Analyze Sensitive Data**:
   - Check for plaintext sensitive data such as:
     - Session tokens
     - Personally identifiable information (PII)
     - API keys

4. **Inspect Storage Expiry**:
   - Verify if temporary data (e.g., session-specific tokens) is improperly stored in persistent storage.

---

### Step 4: Validate Exploitation
1. Confirm if tampered or stolen client-side data leads to:
   - Unauthorized access or privilege escalation.
   - Data leakage to unauthorized parties.

2. Assess the impact on the application, such as:
   - Security and privacy violations.
   - Circumvention of application logic.

---

## Tools

- **Browser Developer Tools** for inspecting and modifying client-side storage.
- **Burp Suite** or **OWASP ZAP** for analyzing cookies and storage mechanisms.
- **Custom Scripts** (e.g., JavaScript in browser consoles) for automated tampering:
   ```javascript
   localStorage.setItem('role', 'admin');
   document.cookie = "sessionid=malicious_value";
```

---

## Remediation

1. **Avoid Storing Sensitive Data on the Client**:
    
    - Store sensitive data like session tokens or passwords on the server instead.
2. **Encrypt Data**:
    
    - Encrypt any sensitive data stored on the client using secure encryption algorithms.
3. **Implement Server-Side Validation**:
    
    - Never rely solely on client-side data for critical application logic.
    - Validate all client-provided data on the server before processing.
4. **Use Secure Cookies**:
    
    - Set the `Secure`, `HttpOnly`, and `SameSite` attributes for cookies to prevent client-side access and mitigate attacks.
5. **Limit Storage Usage**:
    
    - Store only non-sensitive, temporary data in client-side storage.
6. **Regularly Audit Storage Practices**:
    
    - Periodically review client-side storage for compliance with security best practices.

---

## References

- [OWASP Testing Guide - Client-Side Storage Testing](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A03:2021 Injection](https://owasp.org/Top10/A03_2021-Injection/)
- [MDN Web Docs - Client-Side Storage](https://developer.mozilla.org/en-US/docs/Web/API/Storage)




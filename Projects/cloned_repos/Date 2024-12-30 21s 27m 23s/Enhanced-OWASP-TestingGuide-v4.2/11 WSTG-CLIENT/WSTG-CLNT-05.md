# WSTG-CLNT-05: Testing for Cross-Site Flashing (CSF)

## Summary

Cross-Site Flashing (CSF) vulnerabilities occur when a malicious website can exploit a trusted website's Flash content to execute unauthorized actions or steal sensitive data. Although Flash is now largely deprecated, legacy applications that still use Flash content may remain vulnerable to CSF attacks.

## Objective

To identify and exploit scenarios where Flash content in a trusted application can be misused to:

- Perform unauthorized actions on behalf of a victim.
- Steal sensitive information such as cookies, tokens, or credentials.

## How to Test

### Step 1: Identify Flash Content
1. Locate Flash (.SWF) files in the application by inspecting:
   - Embedded Flash elements in the application.
   - URLs or endpoints serving Flash files.

2. Analyze the parameters or inputs accepted by the Flash files:
   - Query parameters
   - POST data
   - URL fragments

---

### Step 2: Review Flash Content
1. Decompile the Flash content using tools like **JPEXS Free Flash Decompiler** or **SWF Investigator** to analyze its functionality.
2. Identify unsafe actions, such as:
   - Handling user-controlled input without validation.
   - Dynamically loading external content or scripts.
   - Performing sensitive operations without proper authentication.

---

### Step 3: Perform CSF Testing
1. **Inject Malicious Parameters**:
   - Modify parameters passed to the Flash file to test for unintended behaviors.
   - Example: Inject JavaScript code or unauthorized URLs.

2. **Simulate Cross-Origin Interactions**:
   - Test if a malicious website can load the Flash file and pass inputs.
   - Example: Embed the trusted Flash file on a malicious site and pass custom parameters.

3. **Analyze Data Leakage**:
   - Verify if sensitive data (e.g., cookies, tokens) is sent to the malicious site.
   - Use tools like Burp Suite or OWASP ZAP to intercept and analyze requests.

---

### Step 4: Exploit CSF Vulnerabilities
1. Craft a proof-of-concept (PoC) that demonstrates:
   - Unauthorized actions performed using the Flash content.
   - Sensitive data exfiltration to an attacker's server.

2. Validate the impact on:
   - Confidentiality (e.g., data leakage).
   - Integrity (e.g., unauthorized actions).
   - Availability (e.g., disruption caused by malicious inputs).

---

## Tools

- **JPEXS Free Flash Decompiler** for analyzing Flash files
- **SWF Investigator** for inspecting Flash content
- **Burp Suite** or **OWASP ZAP** for intercepting and manipulating requests
- **Custom Scripts** for simulating cross-origin requests or parameter injections

---

## Remediation

1. **Migrate Away from Flash**:
   - Replace Flash content with modern technologies such as HTML5, which is more secure and widely supported.

2. **Validate All Inputs**:
   - Implement strict validation for inputs processed by Flash files to prevent malicious manipulation.

3. **Enforce Cross-Origin Resource Sharing (CORS)**:
   - Restrict cross-origin interactions to trusted domains only.

4. **Use Secure Authentication Mechanisms**:
   - Ensure that sensitive operations require proper authentication and cannot be executed via Flash content alone.

5. **Disable Flash Support**:
   - If Flash is not necessary, remove it entirely to eliminate the attack surface.

---

## References

- [OWASP Testing Guide - Testing for Flash Security](https://owasp.org/www-project-testing/)
- [Adobe Security - Best Practices for Flash Content](https://helpx.adobe.com/security.html)
- [MDN Web Docs - Flash and Browser Security](https://developer.mozilla.org/)

---

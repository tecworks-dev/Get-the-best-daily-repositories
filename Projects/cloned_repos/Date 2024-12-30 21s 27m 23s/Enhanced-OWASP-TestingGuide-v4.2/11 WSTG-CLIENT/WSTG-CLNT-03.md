# WSTG-CLNT-03: Testing for Client-Side URL Redirects

## Summary

Client-side URL redirect vulnerabilities occur when an application allows user input to control the destination of a URL redirection. This can be exploited by attackers to redirect users to malicious websites, leading to phishing attacks, malware distribution, or other malicious activities.

## Objective

To identify and exploit scenarios where user-controlled inputs can manipulate URL redirection logic on the client side, causing unauthorized or malicious redirects.

## How to Test

### Step 1: Identify Potential Redirect Points
1. Locate areas in the application where redirection occurs, such as:
   - Links or buttons that redirect users based on input.
   - Query parameters (e.g., `?redirect=example.com`).
   - Fragments (e.g., `#redirect=example.com`).

2. Inspect JavaScript code that handles redirection using:
   - `window.location`
   - `window.location.assign()`
   - `window.location.replace()`
   - `window.open()`

---

### Step 2: Inject Test Payloads
1. Inject payloads into identified inputs that influence redirection, such as:
   - URL query parameters
   - POST data
   - Cookies or localStorage

2. Use test payloads to observe redirection behavior:
   - Valid domain: `https://example.com`
   - External domain: `https://malicious.com`
   - JavaScript payloads: `javascript:alert('XSS')`

3. Test edge cases like:
   - Relative URLs: `//malicious.com`
   - Protocol-less URLs: `//malicious.com/path`
   - Mixed characters: `https://legit.com@malicious.com`

---

### Step 3: Analyze Results
1. Verify if the application redirects to user-supplied destinations without validation.
2. Observe the behavior of JavaScript handling redirects. Examples include:
   - Redirection to external or malicious sites.
   - Execution of `javascript:` URLs.
3. Confirm if the vulnerability can be exploited remotely by crafting malicious links.

---

### Step 4: Exploitability Assessment
1. Test if an attacker can exploit the redirect to:
   - Steal user credentials (phishing).
   - Deliver malware or malicious payloads.
   - Manipulate user trust or behavior.

2. Validate the scope of exploitation, including:
   - Target users (authenticated or unauthenticated).
   - Impact of redirection on user security and privacy.

---

## Tools

- **Browser Developer Tools** for inspecting redirect behavior
- **Burp Suite** or **OWASP ZAP** for testing and manipulating redirection inputs
- **Custom Scripts** (e.g., Python with `requests` or Selenium) for automated testing
- **Fuzzers** to systematically test for unsafe redirects

---

## Remediation

1. **Validate and Sanitize Inputs**:
   - Allow redirection only to whitelisted or internal domains.
   - Reject or sanitize any user input controlling redirect destinations.

2. **Use Absolute URLs for Redirection**:
   - Avoid using relative or user-controlled URLs for redirects.

3. **Implement Warning Messages**:
   - Warn users before redirecting to external domains.

4. **Avoid Using `javascript:` in URLs**:
   - Restrict or disable the use of `javascript:` or other dangerous protocols in redirect logic.

5. **Perform Regular Testing**:
   - Conduct periodic assessments to identify and mitigate redirect vulnerabilities.

---

## References

- [OWASP Testing Guide - Testing for Client-Side URL Redirects](https://owasp.org/www-project-testing/)
- [OWASP Top Ten - A10:2021 Server-Side Request Forgery (SSRF)](https://owasp.org/Top10/A10_2021-SSRF/)
- [Google Safe Browsing - Preventing Open Redirects](https://developers.google.com/web/fundamentals/security/)

---

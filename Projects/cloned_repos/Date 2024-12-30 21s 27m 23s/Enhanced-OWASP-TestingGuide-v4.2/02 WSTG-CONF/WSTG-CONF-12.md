## WSTG-CONF-12: Testing for Content Security Policy

### Objective
Evaluate the application's Content Security Policy (CSP) to ensure it effectively mitigates client-side attacks, such as Cross-Site Scripting (XSS) and data injection attacks.

### Testing Approach

1. **Check CSP Implementation**
   - Verify the presence of the `Content-Security-Policy` HTTP header in responses.
   - Confirm that the CSP is appropriately configured and applied across all pages.
   - Tools: Browser developer tools, Burp Suite, or `curl`.

2. **Evaluate CSP Directives**
   - Analyze CSP directives to ensure they provide sufficient protection:
     - `default-src`: Restrict default content sources.
     - `script-src`: Allow only trusted sources for scripts.
     - `object-src`: Avoid allowing object embedding if not required.
   - Tools: Manual review or CSP evaluation tools like [CSP Evaluator](https://csp-evaluator.withgoogle.com).

3. **Test for Bypass Opportunities**
   - Check if CSP allows inline scripts using `unsafe-inline`.
   - Identify overly permissive directives (e.g., `script-src 'self' 'unsafe-inline'`).
   - Tools: Proxy tools like Burp Suite, automated CSP scanners.

4. **Analyze Reporting Features**
   - Confirm that the `report-uri` or `report-to` directive is set up for CSP violation reporting.
   - Ensure reports are sent to a secure and monitored endpoint.
   - Tools: Monitor CSP violation reports using browser developer tools or third-party services.

5. **Test for Compatibility Issues**
   - Verify CSP does not break the application's functionality.
   - Test with multiple browsers to ensure consistent enforcement.
   - Tools: Browser testing and debugging tools.

### Tools and Resources
- **Browser Developer Tools**
  - Inspect the `Content-Security-Policy` header.
- **CSP Evaluator**
  - Analyze and validate CSP configurations.
- **Burp Suite**
  - Inject malicious payloads to identify CSP bypass opportunities.
- **curl**
  - Example: `curl -I https://example.com`

### Recommendations
- Apply a restrictive CSP that only allows resources from trusted origins.
- Avoid using `unsafe-inline` or `unsafe-eval` directives.
- Use `report-uri` or `report-to` for monitoring CSP violations.
- Regularly audit and update the CSP to align with the application's needs.

---

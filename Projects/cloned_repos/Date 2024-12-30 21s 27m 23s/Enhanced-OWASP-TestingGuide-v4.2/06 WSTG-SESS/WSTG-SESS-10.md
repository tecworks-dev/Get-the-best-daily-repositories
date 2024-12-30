# WSTG-SESS-10 - Testing for Cross-Site Script Inclusion (XSSI)

## Summary
Cross-Site Script Inclusion (XSSI) attacks exploit improperly secured JSON or JavaScript responses by embedding them in malicious web pages. This can lead to unauthorized access to sensitive data, including session tokens or user information.

## Objective
To identify if the application is vulnerable to XSSI attacks by exposing sensitive data in JavaScript or JSON responses.

## Testing Procedure

### 1. Identify Exposed JSON or JavaScript Responses
- **Description**: Locate endpoints that serve JSON or JavaScript data.
- **Steps**:
  1. Use browser developer tools or interception proxies (e.g., Burp Suite, OWASP ZAP) to monitor responses.
  2. Identify API or JavaScript endpoints serving JSON or script content.
  3. Note if any sensitive data, such as session tokens or user information, is included in the response.

### 2. Test Embedding in Malicious Pages
- **Description**: Verify if JSON or JavaScript responses can be exploited through embedding.
- **Steps**:
  1. Create a simple HTML page that includes the target JSON or JavaScript endpoint using `<script>` tags.
  2. Load the page in a browser and observe if sensitive data is exposed in the developer tools console.
  3. Test if sensitive data can be accessed using JavaScript (`window.onload` or `eval`).

### 3. Inspect Content-Type Headers
- **Description**: Confirm if the server specifies correct `Content-Type` headers.
- **Steps**:
  1. Capture responses from the target endpoint.
  2. Verify if the `Content-Type` is correctly set (e.g., `application/json` for JSON).
  3. Test if setting incorrect `Content-Type` allows the content to be treated as JavaScript.

### 4. Test for Data Leakage
- **Description**: Check if sensitive information is returned in JSON or JavaScript responses.
- **Steps**:
  1. Access the JSON or JavaScript endpoint directly in a browser.
  2. Observe the response for sensitive data like session tokens or user details.
  3. Analyze if this data could be used maliciously.

### 5. Check for Anti-XSSI Mechanisms
- **Description**: Verify if the application implements anti-XSSI protections.
- **Steps**:
  1. Check if the response includes anti-XSSI prefixes (e.g., `)]}',` or similar).
  2. Confirm that the prefix prevents direct execution of JSON or JavaScript by browsers.

## Tools
- Burp Suite
- OWASP ZAP
- Browser Developer Tools
- Custom HTML pages for testing inclusion

## Remediation
1. Use anti-XSSI prefixes in JSON responses.
2. Implement proper CORS (Cross-Origin Resource Sharing) policies to restrict access.
3. Ensure sensitive data is not included in JSON or JavaScript responses.
4. Specify correct `Content-Type` headers (e.g., `application/json`).
5. Avoid returning executable JavaScript unless necessary and ensure it does not expose sensitive information.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP Cross-Site Scripting Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html)

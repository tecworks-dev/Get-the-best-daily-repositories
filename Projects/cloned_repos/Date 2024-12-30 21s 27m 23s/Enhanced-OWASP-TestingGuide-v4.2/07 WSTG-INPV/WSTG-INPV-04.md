# WSTG-INPV-04 - Testing for HTTP Parameter Pollution (HPP)

## Summary
HTTP Parameter Pollution (HPP) occurs when an attacker manipulates or injects multiple HTTP parameters with the same name in a single request, potentially bypassing security controls or triggering unexpected behavior. HPP can lead to issues such as authentication bypass, privilege escalation, or denial of service.

## Objective
To identify if the application is vulnerable to HTTP Parameter Pollution by testing the handling of multiple parameters with the same name.

## Testing Procedure

### 1. Identify Input Points
- **Description**: Locate input fields or parameters that the application processes.
- **Steps**:
  1. Use the application to identify query parameters, form fields, or HTTP headers.
  2. Note which parameters are processed by the server.

### 2. Test Query Parameters
- **Description**: Inject multiple parameters with the same name in the URL.
- **Steps**:
  1. Modify the URL to include multiple values for the same parameter (e.g., `?param=value1&param=value2`).
  2. Observe how the application processes these values (e.g., chooses the first, last, or concatenates them).

### 3. Test Form Fields
- **Description**: Submit a form with duplicate field names.
- **Steps**:
  1. Modify the form submission to include multiple fields with the same name.
  2. Analyze how the application processes the duplicate fields.

### 4. Test HTTP Headers
- **Description**: Inject duplicate headers in the HTTP request.
- **Steps**:
  1. Use tools like Burp Suite to add multiple headers with the same name.
  2. Observe the server's response to determine how it processes duplicate headers.

### 5. Test for Security Control Bypass
- **Description**: Attempt to bypass validation or security controls using HPP.
- **Steps**:
  1. Inject valid and invalid values for the same parameter (e.g., `?role=admin&role=user`).
  2. Analyze if security controls are bypassed due to conflicting values.

### 6. Test for Unexpected Behavior
- **Description**: Check if HPP leads to errors or unexpected behavior in the application.
- **Steps**:
  1. Inject random or malicious values for duplicate parameters.
  2. Observe the server's behavior, including error messages or crashes.

## Tools
- Burp Suite
- OWASP ZAP
- Postman
- Browser Developer Tools

## Remediation
1. Validate and sanitize all incoming parameters on the server side.
2. Reject or handle multiple parameters with the same name consistently.
3. Implement server-side logging to detect and investigate suspicious requests.
4. Educate developers on secure coding practices to mitigate HPP.
5. Use frameworks or libraries that provide secure request handling.

## References
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [OWASP HTTP Parameter Pollution](https://owasp.org/www-community/attacks/HTTP_Parameter_Pollution)

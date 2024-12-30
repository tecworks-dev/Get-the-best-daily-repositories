# WSTG-CONF-07: Test HTTP Strict Transport Security (HSTS)

## Objective
Ensure that HTTP Strict Transport Security (HSTS) is correctly implemented to protect users from protocol downgrade attacks and cookie hijacking.

## Checklist

### 1. Verify HSTS Header
- Confirm the presence of the `Strict-Transport-Security` header in HTTPS responses.
  - Example:
    ```
    Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
    ```

### 2. Check for Adequate `max-age` Value
- Ensure `max-age` is set to a sufficient duration (e.g., 1 year = `31536000` seconds).

### 3. Validate `includeSubDomains`
- Verify that the `includeSubDomains` directive is used to enforce HSTS on all subdomains.

### 4. Test Preload Readiness
- Check if the application is enrolled in the HSTS preload list:
  - Verify the `preload` directive is included.
  - Test the domain using the [HSTS Preload Submission](https://hstspreload.org/) website.

### 5. Ensure HTTPS Redirection
- Confirm that HTTP traffic is automatically redirected to HTTPS.
- Verify that all resources (e.g., scripts, images) are served over HTTPS.

### 6. Test Downgrade Attack Mitigation
- Attempt to access the site using HTTP:
  - Ensure the connection is refused or redirected to HTTPS.
- Simulate a downgrade scenario and confirm HSTS prevents access.

### 7. Analyze Error Messages
- Verify that error messages for non-secure connections do not reveal sensitive information.

## Tools
- **cURL**: To test for the presence of HSTS headers (e.g., `curl -I https://example.com`).
- **Burp Suite**: For analyzing headers and testing redirection behavior.
- **SSL Labs**: For comprehensive HTTPS and HSTS configuration testing.
- **HSTS Preload Submission Checker**: To validate preload readiness.

---

Let me know if you'd like any modifications or if you're ready for the next file!

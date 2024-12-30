# WSTG-CONF-06: Test HTTP Methods

## Objective
Identify misconfigured or insecure HTTP methods (e.g., `PUT`, `DELETE`) that could lead to unauthorized actions or data manipulation.

## Checklist

### 1. Enumerate Enabled HTTP Methods
- Use tools to identify supported HTTP methods:
  - `GET`
  - `POST`
  - `HEAD`
  - `OPTIONS`
  - `PUT`
  - `DELETE`
  - `TRACE`
  - `CONNECT`

### 2. Verify Allowed Methods
- Ensure that only required methods (e.g., `GET`, `POST`) are enabled.
- Test for sensitive methods:
  - `PUT`: Check if it allows uploading files.
  - `DELETE`: Test if it enables file deletion.
  - `TRACE`: Validate if the server reflects headers (used in XST attacks).

### 3. Test for Unauthorized Actions
- Attempt to use methods like `PUT` or `DELETE` to:
  - Upload unauthorized files.
  - Delete resources.
  - Modify server configurations.

### 4. Analyze HTTP OPTIONS Response
- Confirm that unnecessary methods are disabled in the server response to `OPTIONS` requests.

### 5. Check for WebDAV Extensions
- If WebDAV is enabled, test for methods such as `PROPFIND`, `MKCOL`, `COPY`, `MOVE`, which may allow unintended actions.

### 6. Validate Authentication for Methods
- Verify that methods like `PUT` and `DELETE` require proper authentication and authorization.
- Ensure error responses (e.g., `401 Unauthorized`, `403 Forbidden`) are correctly returned for unauthorized users.

### 7. Inspect HTTP Headers
- Check for headers that may indicate misconfigurations, such as `Allow` or `Public`.

## Tools
- **cURL**: For manual testing of HTTP methods (e.g., `curl -X METHOD`).
- **Burp Suite**: To craft and test custom HTTP requests.
- **Nmap**: With `http-methods` script for automated enumeration.
- **OWASP ZAP**: For testing method vulnerabilities in an automated fashion.

---

Let me know if you'd like to refine this section or proceed to the next file!

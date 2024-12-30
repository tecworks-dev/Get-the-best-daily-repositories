# WSTG-INFO-06: Identify Application Entry Points

## Objective
To identify all potential entry points in the application, including pages, forms, APIs, and other input mechanisms, that an attacker could target.

## Key Steps

### 1. Identify Visible Entry Points
Manually browse the application to identify visible inputs and functionalities.
- Common entry points:
  - Login forms
  - Search boxes
  - Contact forms
  - File upload features

### 2. Inspect URLs for Parameters
Analyze the application’s URLs to locate parameters that accept user input.
- Examples:
  ```
  http://targetdomain.com/product?id=123
  http://targetdomain.com/search?q=example
  ```

### 3. Enumerate Hidden Entry Points
Identify hidden fields, parameters, and endpoints not visible in the UI.
- Use browser developer tools to inspect:
  - Hidden form fields
  - JavaScript variables
- Check for comments or scripts that reference hidden endpoints.

### 4. Crawl the Application
Automate the discovery of all entry points by crawling the application.
- Tools:
  - [Burp Suite](https://portswigger.net/burp): Crawl and log all endpoints.
  - [OWASP ZAP](https://owasp.org/www-project-zap/): Use the spider feature to discover entry points.

### 5. Test for API Endpoints
Analyze the application for exposed API endpoints:
- Look for `/api/` or similar patterns in URLs.
- Inspect network requests using browser developer tools.

### 6. Inspect JavaScript Files
Review linked JavaScript files to find references to:
- Hidden API endpoints.
- Unused or deprecated functionality.

### 7. Check for Multi-Step Processes
Identify multi-step forms or workflows where data can be submitted at each step.
- Examples:
  - Registration forms
  - Shopping carts

### 8. Document Findings
Log all identified entry points with the following details:
- URL or location.
- Description of the entry point.
- Input methods (e.g., GET/POST parameters, file uploads).

## Tools and Resources
- **Browser Tools**:
  - Developer tools (e.g., Chrome DevTools, Firefox Developer Tools).
- **Tools**:
  - Burp Suite
  - OWASP ZAP
  - [Postman](https://www.postman.com/)

## Mitigation Recommendations
- Validate and sanitize all user inputs from identified entry points.
- Restrict access to unnecessary or unused endpoints.
- Regularly review and test entry points for security vulnerabilities.

---

**Next Steps:**
Proceed to [WSTG-INFO-07: Map Execution Paths Through Application](./WSTG_INFO_07.md).

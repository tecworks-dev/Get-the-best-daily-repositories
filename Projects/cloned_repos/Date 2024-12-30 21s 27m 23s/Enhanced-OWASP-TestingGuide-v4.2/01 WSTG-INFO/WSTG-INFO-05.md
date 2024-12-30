# WSTG-INFO-05: Review Webpage Content for Information Leakage

## Objective
To analyze the content of web pages for sensitive information that might inadvertently expose internal details, credentials, or other exploitable data.

## Key Steps

### 1. Review Visible Content
Manually browse the web pages to identify:
- Sensitive data displayed on pages (e.g., user information, credentials).
- Comments in the HTML source containing internal notes or sensitive details.

### 2. Examine Metadata
Inspect metadata tags and attributes for potential information leakage:
- `meta` tags (e.g., `author`, `generator`).
- Example:
  ```html
  <meta name="generator" content="WordPress 5.9">
  ```

### 3. Search for Hardcoded Credentials
Look for credentials hardcoded in:
- HTML comments.
- JavaScript files included in the pages.
- Inline JavaScript or CSS.

### 4. Analyze Included JavaScript Files
Identify sensitive data or API keys in JavaScript files linked to the webpage.
- Tools:
  - Browser developer tools
  - Command line tools (`wget`, `curl`) to download and inspect JavaScript files.

### 5. Check for Debugging Information
Identify debug output or stack traces accidentally exposed on the web pages.
- Look for error messages or debugging tools left active.

### 6. Inspect Hidden Form Fields
Check for hidden form fields containing sensitive data:
- Example:
  ```html
  <input type="hidden" name="user_id" value="12345">
  ```

### 7. Review Third-Party Content
Analyze embedded content or third-party scripts:
- Review what data is being shared with third parties.
- Inspect the privacy and security policies of third-party services.

### 8. Automate Content Discovery
Use automated tools to crawl and extract content:
- [Burp Suite](https://portswigger.net/burp): Use the crawler to identify sensitive data.
- [Dirb](https://tools.kali.org/web-applications/dirb): Search for hidden files and directories.

### 9. Document Findings
Log all instances of sensitive information leakage:
- URL of the page.
- Description of the sensitive data found.
- Risk assessment and impact analysis.

## Tools and Resources
- **Browser Tools**:
  - Developer tools (e.g., Chrome DevTools, Firefox Developer Tools).
- **Command Line**:
  - `wget`, `curl` for downloading and inspecting files.
- **Tools**:
  - Burp Suite
  - Dirb
  - OWASP ZAP

## Mitigation Recommendations
- Avoid hardcoding sensitive data (e.g., credentials, API keys) in web pages or JavaScript files.
- Regularly audit web pages and source code for unintended disclosures.
- Minimize the use of `hidden` form fields for storing sensitive information.
- Implement robust error handling to suppress debugging or stack trace details.

---

**Next Steps:**
Proceed to [WSTG-INFO-06: Identify Application Entry Points](./WSTG_INFO_06.md).

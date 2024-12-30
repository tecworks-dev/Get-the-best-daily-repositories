# WSTG-INPV-18: Testing for Deserialization Vulnerabilities

## Summary
Deserialization vulnerabilities occur when untrusted data is used to populate objects via deserialization. This can lead to remote code execution (RCE), data tampering, or denial of service (DoS) attacks.

---

## Objectives
- Identify endpoints that deserialize untrusted input.
- Test for deserialization vulnerabilities.
- Assess the potential impact of exploiting deserialization flaws.

---

## How to Test

### Manual Testing

1. **Identify Input Points:**
   - Locate application functionality that accepts serialized data (e.g., cookies, API requests, or hidden form fields).
   - Example serialized data:
     - JSON:
       ```json
       {"user":"admin","role":"user"}
       ```
     - Java serialized object:
       ```
       rO0ABXNyABFqYXZhLnV0aWwuQXJyYXlMaXN0eJyvgZrbZIECABIAAHhw
       ```

2. **Modify Serialized Data:**
   - Inject malicious payloads into serialized data to test for vulnerabilities:
     - Example JSON manipulation:
       ```json
       {"user":"admin","role":"admin"}
       ```
     - Example Java gadget chain:
       ```
       CommonsCollections1 payload
       ```

3. **Analyze Application Behavior:**
   - Check for changes in behavior, such as privilege escalation, unexpected outputs, or errors revealing deserialization traces.
   - Example error:
     ```
     java.io.InvalidClassException
     ```

4. **Test for Remote Code Execution (RCE):**
   - Use known gadget chains to test for RCE in deserialization:
     - Example with `ysoserial`:
       ```bash
       java -jar ysoserial.jar CommonsCollections1 "calc" > payload.ser
       ```
   - Send the serialized payload to the target endpoint.

---

### Automated Testing

Use tools like:
- **Burp Suite Extensions:** Identify deserialization issues by analyzing requests and responses.
- **Serializator:** Detect and test deserialization vulnerabilities.
- **ysoserial:** Generate payloads for known deserialization vulnerabilities.

#### Example with Burp Suite:
- Intercept and modify serialized data in requests.
- Replay modified requests and analyze responses.

---

## Mitigation
- Use secure deserialization practices:
  - Validate and sanitize all serialized inputs.
  - Use cryptographic signatures to verify serialized data integrity.
  - Avoid using native deserialization methods when processing untrusted data.
- Implement allowlist-based object validation during deserialization.
- Use libraries that offer secure deserialization.
- Regularly update and patch deserialization libraries.

---

## References
- OWASP Deserialization Cheat Sheet
- CWE-502: Deserialization of Untrusted Data
- Tools:
  - [ysoserial](https://github.com/frohoff/ysoserial)
  - [Burp Suite Extensions](https://portswigger.net/bappstore)
  - [Serializator](https://github.com/ambionics/serializator)

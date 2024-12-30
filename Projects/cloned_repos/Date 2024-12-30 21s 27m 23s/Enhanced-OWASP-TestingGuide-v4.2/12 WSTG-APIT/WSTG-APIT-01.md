# WSTG-APIT-01: Testing for Web Service Vulnerabilities (RESTful and GraphQL APIs)

## Summary

Web services, including RESTful and GraphQL APIs, are essential for modern application communication. Vulnerabilities in APIs can lead to data leakage, unauthorized access, and application compromise. This guide covers the identification and exploitation of vulnerabilities in both REST and GraphQL APIs.

## Objective

To identify and exploit vulnerabilities in RESTful and GraphQL APIs, including:

- Authentication and authorization issues
- Data leakage
- Input validation vulnerabilities
- Misconfigurations leading to security risks

## How to Test

### Step 1: Enumerate API Endpoints
1. **RESTful API**:
   - Identify endpoints by analyzing:
     - Application documentation or API specifications (e.g., OpenAPI/Swagger).
     - Network traffic using tools like Burp Suite or OWASP ZAP.
     - URL patterns in client-side applications.

   - Map HTTP methods (GET, POST, PUT, DELETE, PATCH) for each endpoint.

2. **GraphQL API**:
   - Locate the GraphQL endpoint (commonly `/graphql`).
   - Test the `Introspection` feature:
     - Submit a `POST` request with the following query to list available types and fields:
       ```graphql
       {
         __schema {
           types {
             name
           }
         }
       }
       ```

---

### Step 2: Analyze Authentication and Authorization
1. **RESTful API**:
   - Test if endpoints enforce authentication:
     - Access endpoints without tokens or credentials.
   - Validate authorization:
     - Test privilege escalation by accessing sensitive endpoints with low-privilege credentials.
     - Modify tokens or API keys.

2. **GraphQL API**:
   - Inspect mutations and queries requiring authentication.
   - Bypass authorization by modifying roles or sending unauthorized queries.
   - Test if the introspection query is accessible without credentials.

---

### Step 3: Test for Input Validation Issues
1. **RESTful API**:
   - Inject malicious payloads:
     - SQL Injection: `{"query": "SELECT * FROM users WHERE id=1 OR '1'='1"}`.
     - XSS: `{"input": "<script>alert('XSS')</script>"}`.
     - Command Injection: `{"cmd": "ls; rm -rf /"}`.
   - Send unexpected data types or lengths to parameters.

2. **GraphQL API**:
   - Inject payloads into queries or mutations:
     - Example SQL Injection query:
       ```graphql
       {
         user(id: "1 OR 1=1") {
           id
           name
         }
       }
       ```
   - Test deeply nested queries to identify denial of service (DoS) vulnerabilities:
     ```graphql
     {
       a {
         b {
           c {
             d {
               e
             }
           }
         }
       }
     }
     ```

---

### Step 4: Test for Data Exposure
1. **RESTful API**:
   - Analyze responses for sensitive information:
     - Authentication tokens, PII, or internal server details.
   - Test enumeration through parameters like `?id=` or `?page=`.

2. **GraphQL API**:
   - Use introspection to discover hidden or sensitive fields.
   - Execute queries to retrieve excessive data:
     ```graphql
     {
       users {
         id
         name
         email
         password
       }
     }
     ```

---

### Step 5: Test for Security Misconfigurations
1. **RESTful API**:
   - Verify HTTP headers for security:
     - Missing `X-Content-Type-Options`, `Strict-Transport-Security`, etc.
   - Ensure endpoints are accessible only over HTTPS.

2. **GraphQL API**:
   - Check if the `Introspection` query is unnecessarily enabled in production.
   - Test for exposed debugging or development tools.

---

### Step 6: Analyze Results
1. Document identified vulnerabilities, including:
   - Endpoints or queries exploited.
   - Payloads and results.
   - Observed impacts, such as data leakage or unauthorized access.

2. Assess the overall impact on confidentiality, integrity, and availability.

---

## Tools

- **For RESTful APIs**:
  - **Burp Suite** or **OWASP ZAP** for traffic interception and manipulation.
  - **Postman** or **Insomnia** for crafting API requests.
  - **Fuzzers** like `ffuf` or `dirb` for endpoint discovery.

- **For GraphQL APIs**:
  - **Altair GraphQL Client**, **GraphiQL**, or **Postman** for running queries and mutations.
  - **GraphQL Voyager** for visualizing schema structures.
  - **Custom Scripts** using Python (`requests`, `graphql-client` libraries) for automation.

---

## Remediation

1. **Authentication and Authorization**:
   - Enforce strong authentication mechanisms like OAuth 2.0.
   - Apply role-based access control (RBAC) for both REST and GraphQL endpoints.

2. **Input Validation**:
   - Validate and sanitize all inputs server-side, rejecting unexpected or malicious inputs.

3. **Limit Data Exposure**:
   - Restrict data returned by queries to only what is necessary.
   - Disable GraphQL introspection in production environments.

4. **Secure Communication**:
   - Enforce HTTPS and set secure headers like `Strict-Transport-Security`.

5. **Rate Limiting and Resource Controls**:
   - Implement rate limiting and recursion depth restrictions to prevent abuse.

6. **Regular Testing**:
   - Periodically test API security to identify and fix vulnerabilities.

---

## References

- [OWASP API Security Project](https://owasp.org/www-project-api-security/)
- [OWASP REST Security Cheat Sheet](https://owasp.org/www-project-cheat-sheets/cheatsheets/REST_Security_Cheat_Sheet.html)
- [OWASP GraphQL Security Cheat Sheet](https://owasp.org/www-project-cheat-sheets/cheatsheets/GraphQL_Security_Cheat_Sheet.html)
- [GraphQL Best Practices](https://graphql.org/learn/best-practices/)

---

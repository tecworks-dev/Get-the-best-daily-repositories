# WSTG-INFO-10: Map Application Architecture

## Objective
To understand the architecture of the target application, including its components, dependencies, and communication patterns, to identify potential vulnerabilities and areas of interest for further testing.

## Key Steps

### 1. Identify Application Components
Break down the application into its major components:
- Frontend (e.g., web interface, mobile apps).
- Backend (e.g., web servers, databases, APIs).
- Third-party integrations (e.g., payment gateways, analytics tools).

### 2. Inspect Communication Channels
Map out communication between components:
- Protocols used (e.g., HTTP, HTTPS, WebSocket).
- Data transfer methods (e.g., JSON, XML, SOAP).
- Identify encrypted and unencrypted channels.

### 3. Analyze Application Layers
Understand how the application layers interact:
- Presentation layer (frontend UI/UX).
- Logic layer (backend business logic, APIs).
- Data layer (databases, file storage).

### 4. Map External Dependencies
Identify external systems or services the application relies on:
- APIs (e.g., REST, GraphQL).
- Cloud services (e.g., AWS, Azure, GCP).
- Third-party SDKs or libraries.

### 5. Review API Endpoints
Analyze the structure and functionality of APIs used by the application:
- Gather endpoint details from browser developer tools or tools like Postman.
- Look for patterns in URLs or payloads that indicate functionality.

### 6. Examine User Roles and Permissions
Map the application's user roles and their access levels:
- Identify role-based access controls (RBAC).
- List operations available to different roles.

### 7. Use Tools to Automate Discovery
Employ tools to assist in mapping the application architecture:
- [Burp Suite](https://portswigger.net/burp): Analyze traffic and interactions.
- [OWASP ZAP](https://owasp.org/www-project-zap/): Spider and analyze application structure.

### 8. Document Findings
Create a clear architectural diagram that includes:
- Application components and their relationships.
- Communication protocols and data flows.
- Key external dependencies and integrations.

## Tools and Resources
- **Browser Tools**:
  - Developer tools (e.g., Chrome DevTools, Firefox Developer Tools).
- **Tools**:
  - Burp Suite
  - OWASP ZAP
  - Postman
  - [Fiddler](https://www.telerik.com/fiddler)

## Mitigation Recommendations
- Secure communication channels with encryption (e.g., HTTPS, TLS).
- Regularly audit third-party dependencies for vulnerabilities.
- Implement strict role-based access controls to enforce least privilege.

---

**Next Steps:**
You have completed the Information Gathering section. Proceed to the next phase of testing based on your workflow.

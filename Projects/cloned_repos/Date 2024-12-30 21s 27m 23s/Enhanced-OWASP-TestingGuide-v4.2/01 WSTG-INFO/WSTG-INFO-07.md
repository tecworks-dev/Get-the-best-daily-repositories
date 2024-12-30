# WSTG-INFO-07: Map Execution Paths Through Application

## Objective
To understand the flow of execution through the application by identifying navigation paths, workflows, and how data is passed between different parts of the application.

## Key Steps

### 1. Analyze User Navigation Paths
Manually browse the application to understand its navigation structure.
- Identify:
  - Main pages and their relationships.
  - Links between pages.
  - Navigation menus and breadcrumbs.

### 2. Identify Multi-Step Processes
Map out workflows that involve multiple steps or stages.
- Examples:
  - User registration processes.
  - Checkout flows in e-commerce sites.
  - Form submissions that span multiple pages.

### 3. Use Automated Crawlers
Employ automated tools to systematically explore the application.
- Tools:
  - [Burp Suite](https://portswigger.net/burp): Use the crawler to map the application.
  - [OWASP ZAP](https://owasp.org/www-project-zap/): Leverage the spider feature.
- Output:
  - List of discovered pages and endpoints.
  - Execution flow diagrams.

### 4. Analyze URL Structures
Inspect URL patterns to understand hierarchical relationships.
- Examples:
  - `/product/view/123`
  - `/user/profile/settings`
- Use parameters or path segments to deduce execution paths.

### 5. Inspect State Management
Understand how the application manages state across execution paths:
- Use tools like browser developer tools to inspect cookies, local storage, and session data.
- Identify how session identifiers or tokens are passed between pages.

### 6. Review JavaScript Logic
Analyze JavaScript files to identify client-side navigation logic or workflows.
- Look for:
  - AJAX requests.
  - Dynamic content loading.
  - Navigation-related logic.

### 7. Map API Calls
Identify API calls made during navigation.
- Use browser developer tools to capture network requests.
- Document APIs and their relationships to different workflows.

### 8. Document Findings
Create a clear map of execution paths:
- Diagrams showing page relationships and workflows.
- Details of how data flows between pages and components.
- Description of critical paths and areas of interest.

## Tools and Resources
- **Browser Tools**:
  - Developer tools (e.g., Chrome DevTools, Firefox Developer Tools).
- **Tools**:
  - Burp Suite
  - OWASP ZAP
  - [Postman](https://www.postman.com/)
  - [Fiddler](https://www.telerik.com/fiddler)

## Mitigation Recommendations
- Review and secure critical execution paths.
- Ensure proper validation and access controls on workflows.
- Regularly test the application for broken or unauthorized paths.

---

**Next Steps:**
Proceed to [WSTG-INFO-08: Fingerprint Web Application Framework](./WSTG_INFO_08.md).

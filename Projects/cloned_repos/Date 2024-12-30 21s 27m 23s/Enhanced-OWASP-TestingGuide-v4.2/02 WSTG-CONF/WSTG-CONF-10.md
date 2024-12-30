# WSTG-CONF-10: Test for Subdomain Takeover

## Objective
Identify and mitigate subdomain takeover vulnerabilities that arise when a DNS entry points to an external service that is no longer in use or properly configured.

## Checklist

### 1. Enumerate Subdomains
- Use tools to identify all subdomains of the target domain.
- Example: `subdomain.example.com`.

### 2. Check DNS Records
- Inspect DNS records for subdomains using:
  - **CNAME** entries pointing to external services (e.g., GitHub Pages, AWS, Heroku).
  - Inactive or unclaimed services.

### 3. Validate Subdomain Availability
- Visit the subdomain in a browser or use a tool to check its status:
  - Look for responses indicating that the service is unclaimed or unused.
  - Example: "No such bucket" or "Domain not found" messages.

### 4. Attempt Subdomain Takeover
- If the subdomain is pointing to an external service, attempt to claim or configure the service:
  - Register the unclaimed resource (e.g., S3 bucket, GitHub repository).

### 5. Review DNS TTL and Configuration
- Verify that DNS entries have an appropriate TTL to allow prompt updates if changes are required.
- Remove unnecessary or unused DNS records.

### 6. Inspect for Service Metadata
- Analyze error messages or page contents for sensitive information or misconfigurations.

## Tools
- **Sublist3r**: For subdomain enumeration.
- **Amass**: For comprehensive DNS enumeration.
- **Dig**: To analyze DNS records (`dig subdomain.example.com`).
- **Burp Suite**: For inspecting responses and error messages.
- **Nuclei**: For detecting known subdomain takeover patterns.

---

Let me know if you'd like any adjustments or if you're ready to move to the next file!

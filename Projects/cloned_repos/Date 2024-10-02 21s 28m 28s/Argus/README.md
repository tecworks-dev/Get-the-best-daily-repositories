
<h1 align="center">
  <a href="">
    <picture>
      <source height="200" media="(prefers-color-scheme: dark)" srcset="https://i.imgur.com/nGEReZh.png">
      <img height="200" alt="Argus" src="https://i.imgur.com/FL0dmHd.png">
    </picture>
  </a>
  <br>
</h1>
<p align="center">
   A Python-based toolkit for  Information Gathering and <br>Reconnaissance
</p>

![screenshot](https://i.imgur.com/BXiJ9bC.gif)

---

## About The Project

Argus is an all-in-one, Python-powered toolkit designed to streamline the process of information gathering and reconnaissance. With a user-friendly interface and a suite of powerful modules, Argus empowers you to explore networks, web applications, and security configurations efficiently and effectively.

Whether you're conducting research, performing security assessments with proper authorization, or just curious about network infrastructures, Argus brings a wealth of information to your fingertips—all in one place.

## ⚠️ WARNING: LEGAL DISCLAIMER

This tool is intended for **educational and ethical use only**. The author is not liable for any illegal use or misuse of this tool. Users are solely responsible for their actions and must ensure they have explicit permission to scan the target systems.

---

## 👀 Screenshots

Take a look at Argus in action:
<p float="left" align="middle">
  <img src="https://i.imgur.com/lS64CUp.png" width="49%">
  <img src="https://i.imgur.com/8VtXyEW.png" width="49%">
</p>
<p float="left" align="middle">
  <img src="https://i.imgur.com/rEIPl2h.png" width="49%">
  <img src="https://i.imgur.com/TVmc8gf.png" width="49%">
</p>
<p float="left" align="middle">
  <img src="https://i.imgur.com/1I6x3Gp.png" width="49%">
  <img src="https://i.imgur.com/9EZqNvK.png" width="49%">
</p>
<p float="left" align="middle">
  <img src="https://i.imgur.com/U4fdPSI.png" width="49%">
  <img src="https://i.imgur.com/LnmykFJ.png" width="49%">
</p>

---

## ⚙️ Installation

To get started with Argus, follow these simple steps:

```bash
git clone https://github.com/jasonxtn/argus.git
cd argus
pip install -r requirements.txt
```

Once installed, you can launch Argus with:

```bash
python argus.py
```

---

## 📖 Usage

Argus offers a rich collection of tools categorized into three main areas:

### Network & Infrastructure Tools

These tools help you gather data about a network, uncovering vital details about servers, IP addresses, DNS records, and more:

1. **Associated Hosts**: Discover domains associated with the target.
2. **DNS Over HTTPS**: Resolve DNS securely via encrypted channels.
3. **DNS Records**: Collect DNS records, including A, AAAA, MX, etc.
4. **DNSSEC Check**: Verify if DNSSEC is properly configured.
5. **Domain Info**: Gather information such as registrar details and expiry dates.
6. **Domain Reputation Check**: Check domain trustworthiness using various reputation sources.
7. **IP Info**: Retrieve geographic and ownership details of an IP address.
8. **Open Ports Scan**: Scan the target for open ports and services.
9. **Server Info**: Extract key server details using various techniques.
10. **Server Location**: Identify the physical location of the server.
11. **SSL Chain Analysis**: Analyze the SSL certificate chain for trustworthiness.
12. **SSL Expiry Alert**: Check SSL certificates for upcoming expiry.
13. **TLS Cipher Suites**: List the supported TLS ciphers on the server.
14. **TLS Handshake Simulation**: Simulate a TLS handshake to check for security issues.
15. **Traceroute**: Trace the path packets take to reach the target.
16. **TXT Records**: Fetch TXT records, often used for verification purposes.
17. **WHOIS Lookup**: Perform WHOIS queries to gather domain ownership details.
18. **Zone Transfer**: Attempt to perform DNS zone transfers.

### Web Application Analysis Tools

These modules focus on understanding the structure and security of web applications:

19. **Archive History**: View the target's history using internet archives.
20. **Broken Links Detection**: Find broken links that may lead to user frustration or security gaps.
21. **Carbon Footprint**: Evaluate the environmental impact of a website.
22. **CMS Detection**: Detect the type of CMS used, like WordPress, Joomla, etc.
23. **Cookies Analyzer**: Analyze cookies for secure attributes and potential privacy issues.
24. **Content Discovery**: Discover hidden directories, files, and endpoints.
25. **Crawler**: Crawl the site to uncover data and map out its structure.
26. **Robots.txt Analyzer**: Analyze the `robots.txt` file for hidden resources.
27. **Directory Finder**: Look for directories that may not be indexed publicly.
28. **Performance Monitoring**: Monitor the website's response time and load performance.
29. **Quality Metrics**: Assess the quality of the site's content and user experience.
30. **Redirect Chain**: Follow redirects to analyze if they're safe or malicious.
31. **Sitemap Parsing**: Extract URLs from the site's sitemap.
32. **Social Media Presence Scan**: Analyze the social media profiles linked to the target.
33. **Technology Stack Detection**: Identify the technologies and frameworks the site uses.
34. **Third-Party Integrations**: Discover any third-party services integrated into the site.

### Security & Threat Intelligence Tools

The security modules in Argus are designed to assess the target's defenses and gather threat intelligence:

35. **Censys Reconnaissance**: Use Censys for in-depth details about the target's assets.
36. **Certificate Authority Recon**: Examine the certificate authority details.
37. **Data Leak Detection**: Check for potential data leaks and sensitive data exposure.
38. **Firewall Detection**: Identify whether a firewall or WAF is protecting the target.
39. **Global Ranking**: Look up the site's global ranking to gauge its popularity.
40. **HTTP Headers**: Extract and evaluate HTTP response headers.
41. **HTTP Security Features**: Check for secure HTTP headers such as HSTS, CSP, etc.
42. **Malware & Phishing Check**: Scan the site for signs of malware and phishing risks.
43. **Pastebin Monitoring**: Search paste sites for leaks associated with the target.
44. **Privacy & GDPR Compliance**: Verify compliance with GDPR and other privacy regulations.
45. **Security.txt Check**: Locate and analyze the `security.txt` file for vulnerability disclosure policies.
46. **Shodan Reconnaissance**: Use Shodan to discover open ports, services, and vulnerabilities.
47. **SSL Labs Report**: Get a detailed SSL/TLS assessment via SSL Labs.
48. **SSL Pinning Check**: Check if SSL pinning is implemented on the site.
49. **Subdomain Enumeration**: Discover subdomains of the target domain.
50. **Subdomain Takeover**: Test whether subdomains are vulnerable to takeover.
51. **VirusTotal Scan**: Check the target's reputation using VirusTotal.

### How to Use Argus

1. Launch Argus from the command line.
2. Enter the tool number you want to use from the main menu.
3. Follow the prompts to enter relevant information.
4. Review the results and adjust your strategy accordingly.

**Example Command:**

```bash
root@argus:~# 1
```
This command initiates the **Associated Hosts** tool.

---


## 🛠 Configuration

Certain modules require API keys to work. Make sure to add any necessary API keys in the `config/settings.py` file before running Argus to unlock full functionality.

---

## 🤝 Contributing

We welcome contributions! Whether it's fixing bugs, adding new features, or improving documentation, your help is appreciated. Fork the repo and submit a pull request .

---
## ⭐️ Show Your Support

If this tool has been helpful to you, please consider giving us a star on GitHub! Your support means a lot to us and helps others discover the project.


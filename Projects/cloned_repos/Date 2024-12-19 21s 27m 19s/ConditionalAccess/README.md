# Modern Conditional Access Baseline 2025

This repository contains a comprehensive set of Conditional Access (CA) policies and PowerShell management tools for Microsoft Entra ID (formerly Azure AD), designed to enhance your organization's security posture while maintaining usability.

## 🙏 Acknowledgments

This project builds upon the excellent work of:
- Kenneth van Surksum ([@kennethvs](https://github.com/kennethvs)) - Original CA baseline policies
- Daniel Chronlund ([@DanielChronlund](https://github.com/DanielChronlund)) - DC Toolbox CA implementations

## 📋 Prerequisites

Before implementing these baselines, ensure:

1. Security Defaults are disabled in your tenant
2. Legacy Per-User MFA is disabled for all users (except unlicensed accounts if necessary)
3. Required licenses are available for your users
4. Basic familiarity with Conditional Access concepts

## 🏗️ Implementation Components

### Required Infrastructure
- 42 Entra ID Groups for inclusion/exclusion management
- 44 Conditional Access policies
- Supporting Intune MAM/APP policies

### Policy Modes
- Most policies are deployed in "Report-only" mode for impact assessment
- Compliance-check policies are set to "Off" mode initially to prevent unexpected authentication prompts

## 🛠️ Recommended Tools

### Policy Deployment
- [Intune Management Tool](https://github.com/Micke-K/IntuneManagement) - For importing and managing policies

### Policy Visualization
- [IdPowerToys](https://idpowertoys.merill.net/) - For visualizing and understanding policy interactions

## 📝 Best Practices

1. **Group-Based Assignment**
   - Always use groups for inclusions/exclusions instead of direct user assignments
   - Enables easier management and automated import via Intune Management Tool

2. **Staged Rollout**
   - Start with policies in report-only mode
   - Use provided PowerShell tools to analyze sign-in logs
   - Assess impact before enabling enforcement

3. **Policy Management**
   - Maintain documentation of policy exceptions
   - Regular review of policy effectiveness
   - Monitor for policy conflicts

## 🚀 Implementation Guide

1. Clone this repository
2. Create required Entra ID groups
3. Import baseline policies using Intune Management Tool
4. Review and customize policies for your environment
5. Use provided PowerShell tools to monitor impact
6. Gradually enable enforcement based on analysis

## 📊 Included Tools

This repository includes PowerShell scripts for:
- Managing user/group assignments
- Analyzing sign-in logs for report-only policies
- Impact assessment reporting
- Policy compliance monitoring

## 🔜 Future Plans

- Enhanced PowerShell tools for sign-in log analysis
- Automated impact assessment reporting
- Additional compliance templates
- Integration with Microsoft Graph API
- Additional baseline policies for specific scenarios

## 📚 Documentation

Detailed documentation for each component is available in the respective folders:
- `/policies` - Baseline CA policies
- `/scripts` - PowerShell management tools
- `/docs` - Implementation guides and best practices

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Support

For issues and feature requests, please use the GitHub issues section.
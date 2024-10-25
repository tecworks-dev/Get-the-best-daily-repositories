// Copyright © 2024 Apple Inc. All Rights Reserved.

// APPLE INC.
// PRIVATE CLOUD COMPUTE SOURCE CODE INTERNAL USE LICENSE AGREEMENT
// PLEASE READ THE FOLLOWING PRIVATE CLOUD COMPUTE SOURCE CODE INTERNAL USE LICENSE AGREEMENT (“AGREEMENT”) CAREFULLY BEFORE DOWNLOADING OR USING THE APPLE SOFTWARE ACCOMPANYING THIS AGREEMENT(AS DEFINED BELOW). BY DOWNLOADING OR USING THE APPLE SOFTWARE, YOU ARE AGREEING TO BE BOUND BY THE TERMS OF THIS AGREEMENT. IF YOU DO NOT AGREE TO THE TERMS OF THIS AGREEMENT, DO NOT DOWNLOAD OR USE THE APPLE SOFTWARE. THESE TERMS AND CONDITIONS CONSTITUTE A LEGAL AGREEMENT BETWEEN YOU AND APPLE.
// IMPORTANT NOTE: BY DOWNLOADING OR USING THE APPLE SOFTWARE, YOU ARE AGREEING ON YOUR OWN BEHALF AND/OR ON BEHALF OF YOUR COMPANY OR ORGANIZATION TO THE TERMS OF THIS AGREEMENT.
// 1. As used in this Agreement, the term “Apple Software” collectively means and includes all of the Apple Private Cloud Compute materials provided by Apple here, including but not limited to the Apple Private Cloud Compute software, tools, data, files, frameworks, libraries, documentation, logs and other Apple-created materials. In consideration for your agreement to abide by the following terms, conditioned upon your compliance with these terms and subject to these terms, Apple grants you, for a period of ninety (90) days from the date you download the Apple Software, a limited, non-exclusive, non-sublicensable license under Apple’s copyrights in the Apple Software to download, install, compile and run the Apple Software internally within your organization only on a single Apple-branded computer you own or control, for the sole purpose of verifying the security and privacy characteristics of Apple Private Cloud Compute. This Agreement does not allow the Apple Software to exist on more than one Apple-branded computer at a time, and you may not distribute or make the Apple Software available over a network where it could be used by multiple devices at the same time. You may not, directly or indirectly, redistribute the Apple Software or any portions thereof. The Apple Software is only licensed and intended for use as expressly stated above and may not be used for other purposes or in other contexts without Apple's prior written permission. Except as expressly stated in this notice, no other rights or licenses, express or implied, are granted by Apple herein.
// 2. The Apple Software is provided by Apple on an "AS IS" basis. APPLE MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS, SYSTEMS, OR SERVICES. APPLE DOES NOT WARRANT THAT THE APPLE SOFTWARE WILL MEET YOUR REQUIREMENTS, THAT THE OPERATION OF THE APPLE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, THAT DEFECTS IN THE APPLE SOFTWARE WILL BE CORRECTED, OR THAT THE APPLE SOFTWARE WILL BE COMPATIBLE WITH FUTURE APPLE PRODUCTS, SOFTWARE OR SERVICES. NO ORAL OR WRITTEN INFORMATION OR ADVICE GIVEN BY APPLE OR AN APPLE AUTHORIZED REPRESENTATIVE WILL CREATE A WARRANTY.
// 3. IN NO EVENT SHALL APPLE BE LIABLE FOR ANY DIRECT, SPECIAL, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, COMPILATION OR OPERATION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 4. This Agreement is effective until terminated. Your rights under this Agreement will terminate automatically without notice from Apple if you fail to comply with any term(s) of this Agreement. Upon termination, you agree to cease all use of the Apple Software and destroy all copies, full or partial, of the Apple Software. This Agreement constitutes the entire understanding of the parties with respect to the subject matter contained herein, and supersedes all prior negotiations, representations, or understandings, written or oral. This Agreement will be governed and construed in accordance with the laws of the State of California, without regard to its choice of law rules.
// You may report security issues about Apple products to product-security@apple.com, as described here: https://www.apple.com/support/security/. Non-security bugs and enhancement requests can be made via https://bugreport.apple.com as described here: https://developer.apple.com/bug-reporting/
// EA1937
// 10/02/2024

//
//  PrivateCloudOSValidator.swift
//  darwinOSBits
//

@_spi(Private) import SecureConfigDB

public enum ConfigSecurityPolicy: String, CaseIterable {
    /// DInitConfig must be validated to ensure secure and private handling of **customer** data.
    case customer
    /// DInitConfig must be validated to ensure secure and private handling of **internal carry / live-on** data.
    case carry
}

struct PrivateCloudOSValidationError: Error, CustomStringConvertible {
    var description: String

    init(_ description: String) {
        self.description = description
    }
}

enum PrivateCloudOSValidatorVersion: Int, CaseIterable {
    case zero = 0
    case one = 1
    case two = 2
    case three = 3
    case four = 4
    case five = 5
    case six = 6
    case seven = 7
    case eight = 8
    case nine = 9
}

/*
 We use UTF8String rather than String to avoid false positives
 when checking if a specified CFPref application ID is allowed
 since UTF8String allows us to compare exact utf8 code units
 of a string.
 See the doc comment in UTF8String.swift for more details.
 */
enum PreferencesRule {
    /* Rule to check if an app ID is allowed wholesale or if certain keys are allowed under that ID
       Note: if an app ID is mapped to an empty set, all keys are considered vacuously allowed */
    case allowOnly(applicationIDs: [UTF8String:Set<UTF8String>], introducedIn: PrivateCloudOSValidatorVersion)
    // Rule for denying a key wholesale, regardless of app ID
    case deny(key: UTF8String, introducedIn: PrivateCloudOSValidatorVersion)
}

protocol PrivateCloudOSValidator {
    var policy:String { get }
    var config:DInitConfig { get }
    var isValid:Bool { get set }
    var logger:Logger { get }
    var requestedVersion:Int { get set }
    var latestApprovedVersion:Int { get set }
    var emittedErrors:[String] { get set }
    
    // ensure validators validate entire config
    mutating func validate() throws
    
    // validation methods for each config key
    func validate(caRoots: DInitCARoots?)
    func validate(cryptexConfig: [DInitCryptexConfig]?)
    func validate(diavloConfig: DInitDiavloConfig?)
    func validate(firewallConfig: DInitFirewallConfig?)
    func validate(installConfig: DInitInstallConfig?)
    func validate(packageConfig: [DInitPackageConfig]?)
    func validate(preferencesConfig: [DInitPreferencesConfig]?)
    func validate(narrativeIdentitiesConfig: [DInitNarrativeIdentitiesConfig]?)
    func validate(resultConfig: DInitResultConfig?)
    func validate(userConfig: DInitUserConfig?)
    func validate(logConfig: DInitLogConfig?)
    func validate(networkConfig: [DInitNetworkConfig]?)
    func validate(networkUplinkMTU: Int?)
    func validate(logText: String?)
    func validate(legacyComputerName: String?)
    func validate(legacyHostName: String?)
    func validate(legacyFQDN: String?)
    func validate(computerName: String?)
    func validate(hostName: String?)
    func validate(localHostName: String?)
    func validate(collectPerfData: Bool?)
    func validate(enableSSH: Bool?)
    func validate(enableSSHPasswordAuthentication: Bool?)
    func validate(rebootAfterSetup: Bool?)
    func validate(USRAfterSetup: DInitUSRPurpose?)
    func validate(lockCryptexes: Bool?)
    func validate(preInitCommands: [String]?)
    func validate(preInitCritical: Bool?)
    func validate(issueDCRT: Bool?)
    func validate(secureConfigParameters: JSON?)
    func validate(usageLabel: String?)
    func validate(retainPreviouslyCachedCryptexesUnsafely: Bool?)
    func validate(cryptexCacheMaxTotalSize: Int?)
    func validate(configSecurityPolicy: String?)
    func validate(configSecurityPolicyVersion: Int?)
    func validate(diagnosticsSubmissionEnabled: Bool?)
    func validate(applyTimeoutArgument: String?)
    func validate(bandwidthLimit: UInt64?)
    func validate(tailspinConfig:DInitTailSpinConfig?)
}

extension PrivateCloudOSValidator {
    mutating func validate() throws {
        // rdar://126501826 (Stop supporting config-security-policy-version)
        guard requestedVersion >= latestApprovedVersion else {
            throw PrivateCloudOSValidationError("config-security-policy-version must be equal to the latest version of PrivateCloudOSValidator: \(latestApprovedVersion)")
        }
        // When a new darwin-init key is added, validation logic must be added even if the new key is permissible under any policy
        for key in DInitConfig.CodingKeys.allCases {
            switch key {
            case .caRoots:
                validate(caRoots: config.caRoots)
            case .cryptexConfig:
                validate(cryptexConfig: config.cryptexConfig)
            case .diavloConfig:
                validate(diavloConfig: config.diavloConfig)
            case .firewallConfig:
                validate(firewallConfig: config.firewallConfig)
            case .installConfig:
                validate(installConfig: config.installConfig)
            case .packageConfig:
                validate(packageConfig: config.packageConfig)
            case .preferencesConfig:
                validate(preferencesConfig: config.preferencesConfig)
            case .narrativeIdentitiesConfig:
                validate(narrativeIdentitiesConfig: config.narrativeIdentitiesConfig)
            case .resultConfig:
                validate(resultConfig: config.resultConfig)
            case .userConfig:
                validate(userConfig: config.userConfig)
            case .logConfig:
                validate(logConfig: config.logConfig)
            case .networkConfig:
                validate(networkConfig: config.networkConfig)
            case .networkUplinkMTU:
                validate(networkUplinkMTU: config.networkUplinkMTU)
            case .logText:
                validate(logText: config.logText)
            case .legacyComputerName:
                validate(legacyComputerName: config.legacyComputerName)
            case .legacyHostName:
                validate(legacyHostName: config.legacyHostName)
            case .legacyFQDN:
                validate(legacyFQDN: config.legacyFQDN)
            case .computerName:
                validate(computerName: config.computerName)
            case .hostName:
                validate(hostName: config.hostName)
            case .localHostName:
                validate(localHostName: config.localHostName)
            case .collectPerfData:
                validate(collectPerfData: config.collectPerfData)
            case .enableSSH:
                validate(enableSSH: config.enableSSH)
            case .enableSSHPasswordAuthentication:
                validate(enableSSHPasswordAuthentication: config.enableSSHPasswordAuthentication)
            case .rebootAfterSetup:
                validate(rebootAfterSetup: config.rebootAfterSetup)
            case .USRAfterSetup:
                validate(USRAfterSetup: config.USRAfterSetup)
            case .lockCryptexes:
                validate(lockCryptexes: config.lockCryptexes)
            case .preInitCommands:
                validate(preInitCommands: config.preInitCommands)
            case .preInitCritical:
                validate(preInitCritical: config.preInitCritical)
            case .issueDCRT:
                validate(issueDCRT: config.issueDCRT)
            case .secureConfigParameters:
                validate(secureConfigParameters: config.secureConfigParameters)
            case .usageLabel:
                validate(usageLabel: config.usageLabel)
            case .retainPreviouslyCachedCryptexesUnsafely:
                validate(retainPreviouslyCachedCryptexesUnsafely: config.retainPreviouslyCachedCryptexesUnsafely)
            case .cryptexCacheMaxTotalSize:
                validate(cryptexCacheMaxTotalSize: config.cryptexCacheMaxTotalSize)
            case .configSecurityPolicy:
                validate(configSecurityPolicy: config.configSecurityPolicy)
            case .configSecurityPolicyVersion:
                validate(configSecurityPolicyVersion: config.configSecurityPolicyVersion)
            case .diagnosticsSubmissionEnabled:
                validate(diagnosticsSubmissionEnabled: config.diagnosticsSubmissionEnabled)
            case .applyTimeoutArgument:
                validate(applyTimeoutArgument: config.applyTimeoutArgument)
            case .bandwidthLimit:
                validate(bandwidthLimit: config.bandwidthLimit)
            case .tailspinConfig:
                validate(tailspinConfig: config.tailspinConfig)
            }
        }
        // isValid will be set to false upon policy violation detection - rather than immediate error throwing - so that we receive helpful errors for every violation in the config
        guard isValid else {
            var issues = ""
            for i in 0..<emittedErrors.count {
                issues = "\(issues)\n\(i + 1). \(emittedErrors[i])"
            }
            throw PrivateCloudOSValidationError("darwin-init config is invalid with respect to \(policy) policy due to the following issues: \(issues)")
        }
    }
}

public class CustomerValidator: PrivateCloudOSValidator {
    var logger = Logger.privateCloudOSValidator
    var policy:String
    var requestedVersion:Int
    var config:DInitConfig
    var isValid:Bool
    var latestApprovedVersion:Int
    var emittedErrors:[String]

    let customerPreferencesRules: [PreferencesRule] = [
        /* Rule for allowing the original list of application IDs and keys in earliest version
         that introduced CFPrefs enforcement */
        .allowOnly(applicationIDs: [
            "com.apple.acdc.cloudmetricsd":[],
            "com.apple.acsi.cloudusagetrackingd":[],
            "com.apple.acsi.deploy-manifest-os-variant":[],
            "com.apple.cloudos":[],
            "com.apple.cloudos.AppleComputeEnsembler":[],
            "com.apple.cloudos.cb_attestationd":[],
            "com.apple.cloudos.cb_configurationd":[],
            "com.apple.cloudos.cb_jobauthd":[],
            "com.apple.cloudos.cb_jobhelper":[],
            "com.apple.cloudos.cloudOSInfo":[],
            "com.apple.cloudos.cloudboardd":[],
            "com.apple.cloudos.CloudBoardNullApp":[],
            "com.apple.cloudos.NullCloudController":[],
            "com.apple.cloudos.hotproperties.cb_jobhelper":[],
            "com.apple.cloudos.hotproperties.cloudboardd":[],
            "com.apple.cloudos.hotproperties.test":[],
            "com.apple.cloudos.hotproperties.tie":[],
            "com.apple.narrative":[],
            "com.apple.prcos.splunkloggingd":[],
            "com.apple.privateCloudCompute":[],
            "com.apple.thimble.inference.tie-controllerd":[],
        ], introducedIn: .three),
        // Rule for denying "TiePreferences" key, for backwards compatibility
        .deny(key: "TiePreferences", introducedIn: .three),
        /* Rule for allow list of keys under "com.apple.thimble.inference.tie-controllerd" app ID
         Note: no keys currently allowed under this ID in customer */
        .allowOnly(applicationIDs: [
            "com.apple.acdc.cloudmetricsd":[],
            // com.apple.acsi.cloudusagetrackingd no longer allowed as it is stale
            // com.apple.acsi.deploy-manifest-os-variant no longer allowed as it is stale
            "com.apple.cloudos":[],
            "com.apple.cloudos.AppleComputeEnsembler":[],
            "com.apple.cloudos.cb_attestationd":[],
            "com.apple.cloudos.cb_configurationd":[],
            "com.apple.cloudos.cb_jobauthd":[],
            "com.apple.cloudos.cb_jobhelper":[],
            "com.apple.cloudos.cloudOSInfo":[],
            "com.apple.cloudos.cloudboardd":[],
            "com.apple.cloudos.CloudBoardNullApp":[],
            "com.apple.cloudos.NullCloudController":[],
            "com.apple.cloudos.hotproperties.cb_jobhelper":[],
            "com.apple.cloudos.hotproperties.cloudboardd":[],
            "com.apple.cloudos.hotproperties.test":[],
            "com.apple.cloudos.hotproperties.tie":[],
            // com.apple.narrative no longer allowed as it is stale
            "com.apple.prcos.splunkloggingd":[],
            "com.apple.privateCloudCompute":[],
            // "com.apple.thimble.inference.tie-controllerd" no longer allowed wholesale
        ], introducedIn: .nine),
    ]
    
    init(policy: String, requestedVersion: Int?, config: DInitConfig) {
        self.policy = policy
        // The latested approved validator version is 8. Please reach out to darwinOS team
        // before adding new validation logic and increasing latestApprovedVersion
        self.latestApprovedVersion = PrivateCloudOSValidatorVersion.eight.rawValue
        // unless client requests to override, use latest version
        self.requestedVersion = requestedVersion ?? latestApprovedVersion
        self.config = config
        // Assume the config is valid until first error is found
        self.isValid = true
        self.emittedErrors = []
    }
    
    func emitMessage(_ msg: String) {
        logger.error("\(msg)")
        // Also print the log message for validate subcommand
        print(msg)
        // Save the message to be written to .DarwinSetupStatus later
        emittedErrors.append(msg)
    }
    
    func emitWarning(key: String, subConfig: String?, explanation: String) {
        let prefix = policy.uppercased() + " POLICY VIOLATION WARNING:"
        let context = (subConfig != nil) ? " in \(subConfig!) config" : ""
        let suffix = "in later versions of PrivateCloudOSValidator"
        emitMessage("\(prefix) Invalid \(key) setting\(context). \(explanation) \(suffix)")
    }
    
    // general validation error handling for any invalid config key setting
    func validationError(key: String, subConfig: String? = nil, explanation: String, introducedInVersion: PrivateCloudOSValidatorVersion) {
        // If client wants to fall back to older version, just emit a warning but don't enforce
        guard self.requestedVersion >= introducedInVersion.rawValue else {
            emitWarning(key: key, subConfig: subConfig, explanation: explanation)
            return
        }
        logger.info("Latest version: \(self.latestApprovedVersion)")
        let prefix = policy.uppercased() + " POLICY VIOLATION:"
        let context = (subConfig != nil) ? " in \(subConfig!) config" : ""
        emitMessage("\(prefix) Invalid \(key) setting\(context). \(explanation)")
        isValid = false
    }
    
    // For validating SecureConfigParameters logPolicyPath, returns required path
    func isValidLogPolicyPath(path: String?) -> Bool {
        return path == kCustomerLogPolicyPath
    }
    
    func getValidLogPolicyPaths() -> String {
        return kCustomerLogPolicyPath
    }
    
    func formatAllowedPreferencesIDs(allowed: [UTF8String:Set<UTF8String>]) -> String {
        var identities: [String] = []
        for identity in allowed.keys {
            identities.append("\"\(identity.value)\"")
        }
        return identities.joined(separator: ", ")
    }
    
    func formatAllowedPreferencesKeys(allowed: Set<UTF8String>) -> String {
        var keys: [String] = []
        for key in allowed {
            keys.append("\"\(key.value)\"")
        }
        return keys.joined(separator: ", ")
    }
    
    func getPreferencesRules() -> [PreferencesRule] {
        return customerPreferencesRules
    }
    
    // Diavlo is not supported so this cannot be set
    func validate(diavloConfig: DInitDiavloConfig?) {
        let key = DInitConfig.CodingKeys.diavloConfig.rawValue
        if diavloConfig != nil {
            validationError(key: key, explanation: "Diavlo personalization of cryptexes not supported", introducedInVersion: .two)
        }
    }
    
    func validate(installConfig: DInitInstallConfig?) {
        let key = DInitConfig.CodingKeys.installConfig.rawValue
        if installConfig != nil {
            validationError(key: key, explanation: "Installing custom roots not permitted", introducedInVersion: .two)
        }
    }
    
    func validate(packageConfig: [DInitPackageConfig]?) {
        let key = DInitConfig.CodingKeys.packageConfig.rawValue
        if packageConfig != nil {
            validationError(key: key, explanation: "Installing packages not permitted", introducedInVersion: .two)
        }
    }
    
    func validate(preferencesConfig: [DInitPreferencesConfig]?) {
        guard let preferencesConfig else { return }
        
        let prefsKey = DInitConfig.CodingKeys.preferencesConfig.rawValue
        let appIDKey = DInitPreferencesConfig.CodingKeys.applicationId.rawValue
        let keyKey = DInitPreferencesConfig.CodingKeys.key.rawValue
        
        for config in preferencesConfig {
            let id = config.applicationId ?? kCFPreferencesAnyApplication as String
            let key = config.key
            // Check that each preference doesn't break rules for different versions
            let rules = getPreferencesRules()
            for rule in rules {
                switch rule {
                case .allowOnly(applicationIDs: let appIDs, introducedIn: let version):
                    let allowedKeys = appIDs[UTF8String(stringLiteral: id)]
                    // If app ID is not specified in allowed map at all, it is not allowed wholesale, regardless of key
                    if allowedKeys == nil {
                        validationError(key: appIDKey, subConfig: prefsKey, explanation: "\(id) not allowed. Setting CFPrefs only permitted for the following application IDs: \(formatAllowedPreferencesIDs(allowed: appIDs))", introducedInVersion: version)
                        
                    }
                    // If this allowed app ID has empty keys set, assume all keys are vacuously allowed under this ID
                    // If app ID has non-empty allowed list of keys, deny any key not in that list
                    else if !allowedKeys!.isEmpty && !allowedKeys!.contains(UTF8String(stringLiteral: key)) {
                        validationError(key: appIDKey, subConfig: prefsKey, explanation: "\(key) not allowed. Only the following CFPrefs keys may be set under application ID \(id): \(formatAllowedPreferencesKeys(allowed: allowedKeys!))", introducedInVersion: version)
                    }
                case .deny(key: let deniedKey, introducedIn: let version):
                    // Deny any keys that are not allowed regardless of app ID
                    // Needed for backwards compatibility with earlier versions that allow tie-controllerd wholesale
                    if UTF8String(stringLiteral: key) == deniedKey {
                        validationError(key: keyKey, subConfig: prefsKey, explanation: "Setting CFPrefs key \(key) not permitted", introducedInVersion: version)
                    }
                }
            }
        }
    }
    
    // TODO: ask about narrative options
    func validate(narrativeIdentitiesConfig: [DInitNarrativeIdentitiesConfig]?) {}
    
    func validate(resultConfig: DInitResultConfig?) {
        let resultKey = DInitConfig.CodingKeys.resultConfig.rawValue
        let actionKey = DInitResultConfig.CodingKeys.failureAction.rawValue
        
        // If a result config is not supplied at all, this is valid as .exit is the default
        guard let resultConfig else { return }
        
        // failure action cannot be shutdown but reboot and exit are permitted
        // exit is needed for fleet management system
        switch resultConfig.failureAction {
        case .exit, .reboot:
            break
        case .shutdown:
            validationError(key: actionKey, subConfig: resultKey, explanation: "\(actionKey) must be set to reboot or exit", introducedInVersion: .two)
        }
    }
    
    func validate(userConfig: DInitUserConfig?) {
        let key = DInitConfig.CodingKeys.userConfig.rawValue
        if userConfig != nil {
            validationError(key: key, explanation: "Creating and configuring users not permitted", introducedInVersion: .three)
        }
    }
    
    func validate(logConfig: DInitLogConfig?) {
        let logKey = DInitConfig.CodingKeys.logConfig.rawValue
        let enablementKey = DInitLogConfig.CodingKeys.systemLoggingEnabled.rawValue
        let privacyKey = DInitLogConfig.CodingKeys.systemLogPrivacyLevel.rawValue
        
        guard let logConfig else {
            validationError(key: logKey, explanation: "Private logs must be redacted by setting \(privacyKey) to Public. Logging to disk and snapshots must be disabled by setting \(enablementKey) to false", introducedInVersion: .five)
            return
        }
        
        for key in DInitLogConfig.CodingKeys.allCases {
            switch key {
            case .systemLogPrivacyLevel:
                if logConfig.systemLogPrivacyLevel != DInitSystemLogPrivacyLevel.public {
                    validationError(key: privacyKey, subConfig: logKey, explanation: "Private logs must be redacted by setting \(privacyKey) to Public", introducedInVersion: .five)
                }
            case .systemLoggingEnabled:
                if logConfig.systemLoggingEnabled != false {
                    validationError(key: enablementKey, subConfig: logKey, explanation: "Logging to disk and snapshots must be disabled by setting \(enablementKey) to false", introducedInVersion: .five)
                }
            }
        }
    }
    
    func validate(networkConfig: [DInitNetworkConfig]?) {
        let key = DInitConfig.CodingKeys.networkConfig.rawValue
        if networkConfig != nil {
            validationError(key: key, explanation: "Applying custom network configuration not permitted", introducedInVersion: .two)
        }
    }

    func validate(tailspinConfig:DInitTailSpinConfig?) {
		let key = DInitTailSpinConfig.CodingKeys.tailspin_enabled.rawValue
		if tailspinConfig != nil {
			validationError(key: key, explanation: "Applying custom tailspin configuration is not permitted", introducedInVersion: .two)
		}
    }

    func validate(enableSSH: Bool?) {
        let key = DInitConfig.CodingKeys.enableSSH.rawValue
        if enableSSH == true {
            validationError(key: key, explanation: "Enabling ssh daemon not permitted. \(key) must be unset or false", introducedInVersion: .two)
        }
    }
    
    func validate(enableSSHPasswordAuthentication: Bool?) {
        let key = DInitConfig.CodingKeys.enableSSHPasswordAuthentication.rawValue
        if enableSSHPasswordAuthentication == true {
            validationError(key: key, explanation: "Enabling ssh password authentication not permitted. \(key) must be unset or false", introducedInVersion: .two)
        }
    }
    
    func validate(rebootAfterSetup: Bool?) {
        let key = DInitConfig.CodingKeys.rebootAfterSetup.rawValue
        // darwin-init
        if rebootAfterSetup == true {
            validationError(key: key, explanation: "System reboot after set up not permitted. \(key) must be unset or false", introducedInVersion: .two)
        }
    }
    
    func validate(USRAfterSetup: DInitUSRPurpose?) {
        // if userspace-reboot is unset, darwin-init will read device tree
        // and do a reboot into rem by default, so this is fine
        if USRAfterSetup == nil { return }
        
        // in production you cannot opt of entering REM
        let key = DInitConfig.CodingKeys.USRAfterSetup.rawValue
        if USRAfterSetup != .rem {
            validationError(key: key, explanation: "Opting out of \(key) into rem not permitted", introducedInVersion: .one)
        }
    }
    
    func validate(lockCryptexes: Bool?) {
        let key = DInitConfig.CodingKeys.lockCryptexes.rawValue
        if lockCryptexes == false {
            validationError(key: key, explanation: "Disabling cryptex lockdown not permitted", introducedInVersion: .four)
        }
    }

    func validate(preInitCommands: [String]?) {
        let key = DInitConfig.CodingKeys.preInitCommands.rawValue
        if preInitCommands != nil {
            validationError(key: key, explanation: "Executing pre-init scripts not permitted. \(key) must be unset", introducedInVersion: .two)
        }
    }
    
    func validate(preInitCritical: Bool?) {
        let key = DInitConfig.CodingKeys.preInitCritical.rawValue
        if preInitCritical == true {
            validationError(key: key, explanation: "Executing pre-init scripts not permitted and therefore cannot be critical. \(key) must be unset or false", introducedInVersion: .two)
        }
    }
    
    func validate(secureConfigParameters: JSON?) {
        let secureConfigKey = DInitConfig.CodingKeys.secureConfigParameters.rawValue
        let logPolicyKey = SecureConfigParameters.Keys.logPolicyPath.rawValue
        let metricsKey = SecureConfigParameters.Keys.metricsFilteringEnforced.rawValue
        let logFilteringKey = SecureConfigParameters.Keys.logFilteringEnforced.rawValue
        let crashKey = SecureConfigParameters.Keys.crashRedactionEnabled.rawValue
        let tie_allowNonProdExceptionOptions = SecureConfigParameters.Keys.tie_allowNonProdExceptionOptions.rawValue
        let appleInfrastrucutureEnforcementKey = SecureConfigParameters.Keys.research_disableAppleInfrastrucutureEnforcement.rawValue
        let klogPolicyPaths = getValidLogPolicyPaths()

        if secureConfigParameters == nil {
            validationError(key: secureConfigKey, explanation: "\(secureConfigKey) must be set with \(logPolicyKey) = \(klogPolicyPaths), \(metricsKey) = true, \(logFilteringKey) = true, and \(crashKey) = true", introducedInVersion: .two)
            return
        }
        
        var parameters:SecureConfigParameters
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(secureConfigParameters)
            parameters = try SecureConfigParameters.decode(parametersJson: data, securityPolicy: self.policy)
        } catch {
            validationError(key: secureConfigKey, explanation: "Failed to parse \(secureConfigKey) parameters with error: \(error)", introducedInVersion: .two)
            return
        }
        
        for key in SecureConfigParameters.Keys.allCases {
            switch key {
            case .crashRedactionEnabled:
                if parameters.crashRedactionEnabled != true {
                    validationError(key: crashKey, subConfig: secureConfigKey, explanation: "\(crashKey) must be set to true", introducedInVersion: .two)
                }
            case .internalRequestOptionsAllowed:
                logger.debug("com.apple.tie.internalRequestOptionsAllowed is deprecated, please replace with com.apple.tie.allowNonProdExceptionOptions moving forward.")
                if parameters.internalRequestOptionsAllowed == true {
                    validationError(key: tie_allowNonProdExceptionOptions, explanation: "com.apple.tie.internalRequestOptionsAllowed must be unset or set to false", introducedInVersion: .eight)
                }
            case .tie_allowNonProdExceptionOptions:
                if parameters.tie_allowNonProdExceptionOptions == true {
                    validationError(key: tie_allowNonProdExceptionOptions, explanation: "com.apple.tie.allowNonProdExceptionOptions must be unset or set to false", introducedInVersion: .eight)
                }
            case .logFilteringEnforced:
                if parameters.logFilteringEnforced != true {
                    validationError(key: logFilteringKey, subConfig: secureConfigKey, explanation: "\(logFilteringKey) must be set to true", introducedInVersion: .two)
                }
            case .logPolicyPath:
                if !isValidLogPolicyPath(path: parameters.logPolicyPath) {
                    validationError(key: logPolicyKey, subConfig: secureConfigKey, explanation: "\(logPolicyKey) must be set to \(klogPolicyPaths)", introducedInVersion: .two)
                }
            case .metricsFilteringEnforced:
                if parameters.metricsFilteringEnforced != true {
                    validationError(key: metricsKey, subConfig: secureConfigKey, explanation: "\(metricsKey) must be set to true", introducedInVersion: .two)
                }
            case .research_disableAppleInfrastrucutureEnforcement:
                if parameters.research_disableAppleInfrastrucutureEnforcement == true {
                    if Computer.isVM() {
                        logger.debug("Running in a VM, so \(appleInfrastrucutureEnforcementKey) is valid.")
                    } else {
                        validationError(key: appleInfrastrucutureEnforcementKey, subConfig: secureConfigKey, explanation: "\(appleInfrastrucutureEnforcementKey) must be set to false unless in a VM", introducedInVersion: .eight)
                    }
                }
            @unknown default:
                validationError(key: key.rawValue, subConfig: secureConfigKey, explanation: "Unknown key passed into validator, switch case needs to be updated", introducedInVersion: .one)
            }
        }
    }
    
    func validate(retainPreviouslyCachedCryptexesUnsafely: Bool?) {
        let key = DInitConfig.CodingKeys.retainPreviouslyCachedCryptexesUnsafely.rawValue
        if retainPreviouslyCachedCryptexesUnsafely == true {
            validationError(key: key, explanation: "Retaining cryptexes from previous boot session configs not permitted. \(key) must be unset or false", introducedInVersion: .two)
        }
    }
    
    func validate(cryptexCacheMaxTotalSize: Int?) {
        let key = DInitConfig.CodingKeys.cryptexCacheMaxTotalSize.rawValue
        if cryptexCacheMaxTotalSize != nil {
            validationError(key: key, explanation: "Setting maximum storage space for cryptex cache not permitted. \(key) must be unset", introducedInVersion: .two)
        }
    }
    
    func validate(diagnosticsSubmissionEnabled: Bool?) {
        let key = DInitConfig.CodingKeys.diagnosticsSubmissionEnabled.rawValue
        if diagnosticsSubmissionEnabled == true {
            validationError(key: key, explanation: "Enabling diagnostic log submission not permitted. \(key) must be unset or false", introducedInVersion: .seven)
        }
    }
    
    // allow any setting on customer or carry
    func validate(caRoots: DInitCARoots?) {}
    func validate(cryptexConfig: [DInitCryptexConfig]?) {}
    func validate(firewallConfig: DInitFirewallConfig?) {}
    func validate(networkUplinkMTU: Int?) {}
    func validate(logText: String?) {}
    func validate(legacyComputerName: String?) {}
    func validate(legacyHostName: String?) {}
    func validate(legacyFQDN: String?) {}
    func validate(computerName: String?) {}
    func validate(hostName: String?) {}
    func validate(localHostName: String?) {}
    func validate(collectPerfData: Bool?) {}
    func validate(issueDCRT: Bool?) {}
    func validate(usageLabel: String?) {}
    func validate(applyTimeoutArgument: String?) {}
    func validate(bandwidthLimit: UInt64?) {}
    
    // Only used for validation and not applied
    func validate(configSecurityPolicy: String?) {}
    func validate(configSecurityPolicyVersion: Int?) {}
}

public class CarryValidator: CustomerValidator {
    let carryPreferencesRules: [PreferencesRule] = [
        /* Rule for allowing the original list of application IDs and keys in earliest version
         that introduced CFPrefs enforcement */
        .allowOnly(applicationIDs: [
            "com.apple.acdc.cloudmetricsd":[],
            "com.apple.acsi.cloudusagetrackingd":[],
            "com.apple.acsi.deploy-manifest-os-variant":[],
            "com.apple.cloudos":[],
            "com.apple.cloudos.AppleComputeEnsembler":[],
            "com.apple.cloudos.cb_attestationd":[],
            "com.apple.cloudos.cb_configurationd":[],
            "com.apple.cloudos.cb_jobauthd":[],
            "com.apple.cloudos.cb_jobhelper":[],
            "com.apple.cloudos.cloudOSInfo":[],
            "com.apple.cloudos.cloudboardd":[],
            "com.apple.cloudos.CloudBoardNullApp":[],
            "com.apple.cloudos.NullCloudController":[],
            "com.apple.cloudos.hotproperties.cb_jobhelper":[],
            "com.apple.cloudos.hotproperties.cloudboardd":[],
            "com.apple.cloudos.hotproperties.test":[],
            "com.apple.cloudos.hotproperties.tie":[],
            "com.apple.narrative":[],
            "com.apple.prcos.splunkloggingd":[],
            "com.apple.privateCloudCompute":[],
            "com.apple.thimble.inference.tie-controllerd":[],
            "com.apple.security":["AppleServerAuthenticationAllowUAT"],
        ], introducedIn: .three),
        // Rule for denying "TiePreferences" key, for backwards compatibility
        .deny(key: "TiePreferences", introducedIn: .three),
        // Rule for allow list of keys under "com.apple.thimble.inference.tie-controllerd" app ID
        .allowOnly(applicationIDs: [
            "com.apple.acdc.cloudmetricsd":[],
            // com.apple.acsi.cloudusagetrackingd no longer allowed as it is stale 
            // com.apple.acsi.deploy-manifest-os-variant no longer allowed as it is stale
            "com.apple.cloudos":[],
            "com.apple.cloudos.AppleComputeEnsembler":[],
            "com.apple.cloudos.cb_attestationd":[],
            "com.apple.cloudos.cb_configurationd":[],
            "com.apple.cloudos.cb_jobauthd":[],
            "com.apple.cloudos.cb_jobhelper":[],
            "com.apple.cloudos.cloudOSInfo":[],
            "com.apple.cloudos.cloudboardd":[],
            "com.apple.cloudos.CloudBoardNullApp":[],
            "com.apple.cloudos.NullCloudController":[],
            "com.apple.cloudos.hotproperties.cb_jobhelper":[],
            "com.apple.cloudos.hotproperties.cloudboardd":[],
            "com.apple.cloudos.hotproperties.test":[],
            "com.apple.cloudos.hotproperties.tie":[],
            // com.apple.narrative no longer allowed as it is stale
            "com.apple.prcos.splunkloggingd":[],
            "com.apple.privateCloudCompute":[],
            // "com.apple.thimble.inference.tie-controllerd" no longer allowed wholesale
            "com.apple.thimble.inference.tie-controllerd":["RoutingLayerNameOverride"],
            "com.apple.security":["AppleServerAuthenticationAllowUAT"],
        ], introducedIn: .nine),
    ]
    
    override func isValidLogPolicyPath(path: String?) -> Bool {
        return path == kCustomerLogPolicyPath || path == kCarryLogPolicyPath
    }
    
    override func getValidLogPolicyPaths() -> String {
        return kCarryLogPolicyPath + " or " + kCustomerLogPolicyPath
    }
    
    override func getPreferencesRules() -> [PreferencesRule] {
        return carryPreferencesRules
    }
    
    // Allow configuring users, but disallow setting password and authorized ssh key
    override func validate(userConfig: DInitUserConfig?) {
        guard let userConfig else { return }
        
        let userConfigKey = DInitConfig.CodingKeys.userConfig.rawValue
        let authKey = DInitUserConfig.CodingKeys.sshAuthorizedKeys.rawValue
        let passKey = DInitUserConfig.CodingKeys.password.rawValue

        
        for key in DInitUserConfig.CodingKeys.allCases {
            switch key {
            case .sshAuthorizedKeys:
                if userConfig.sshAuthorizedKeys != nil {
                    validationError(key: authKey, subConfig: userConfigKey, explanation: "Supplying ssh authorized keys not permitted. \(authKey) must be unset", introducedInVersion: .three)
                }
            case .password:
                if userConfig.password != nil {
                    validationError(key: passKey, subConfig: userConfigKey, explanation: "Setting user password not permitted. \(passKey) must be unset", introducedInVersion: .three)
                }
            case .userName, .uid, .gid, .isAdmin, .passwordlessSudo, .appleConnectSSHConfig, .appleAuthenticationConfig:
                break
            }
        }
    }
    
    // Keeping logging to disk enabled and specifying any log preferences permitted for debugging purposes on carry
    override func validate(logConfig: DInitLogConfig?) {
        let logKey = DInitConfig.CodingKeys.logConfig.rawValue
        let privacyKey = DInitLogConfig.CodingKeys.systemLogPrivacyLevel.rawValue
        
        guard let logConfig else {
            validationError(key: logKey, explanation: "Private logs must be redacted by setting \(privacyKey) to Public", introducedInVersion: .five)
            return
        }
        
        for key in DInitLogConfig.CodingKeys.allCases {
            switch key {
            case .systemLogPrivacyLevel:
                if logConfig.systemLogPrivacyLevel != DInitSystemLogPrivacyLevel.public {
                    validationError(key: privacyKey, subConfig: logKey, explanation: "Private logs must be redacted by setting \(privacyKey) to Public", introducedInVersion: .five)
                }
            case .systemLoggingEnabled:
                // Leave logging to disk enabled on carry
                break
            }
        }
    }
    
    // Enabling ssh daemon permitted for debugging purposes on carry
    override func validate(enableSSH: Bool?) {}
    
    override func validate(USRAfterSetup: DInitUSRPurpose?) {
        // if userspace-reboot is unset, darwin-init will read device tree
        // and do a reboot into rem by default, so this is fine
        if USRAfterSetup == nil { return }
        
        let key = DInitConfig.CodingKeys.USRAfterSetup.rawValue
        // in carry version 6 you may only opt into "rem"
        if USRAfterSetup != .rem {
            validationError(key: key, explanation: "Opting out of \(key) into rem not permitted", introducedInVersion: .six)
        }
        
        // in carry versions 1-5 you must at least opt into "rem" or "rem-dev"
        // TODO: remove once self.latestApprovedVersion == 6
        if USRAfterSetup != .remDev && USRAfterSetup != .rem {
            validationError(key: key, explanation: "Opting out of \(key) into rem or rem-dev not permitted", introducedInVersion: .one)
        }
    }
    
    // Allow any setting in carry environments
    override func validate(diagnosticsSubmissionEnabled: Bool?) {}
    
    override func validate(secureConfigParameters: JSON?) {
        let secureConfigKey = DInitConfig.CodingKeys.secureConfigParameters.rawValue
        let logPolicyKey = SecureConfigParameters.Keys.logPolicyPath.rawValue
        let metricsKey = SecureConfigParameters.Keys.metricsFilteringEnforced.rawValue
        let logFilteringKey = SecureConfigParameters.Keys.logFilteringEnforced.rawValue
        let crashKey = SecureConfigParameters.Keys.crashRedactionEnabled.rawValue
        let tie_allowNonProdExceptionOptions = SecureConfigParameters.Keys.tie_allowNonProdExceptionOptions.rawValue
        let appleInfrastrucutureEnforcementKey = SecureConfigParameters.Keys.research_disableAppleInfrastrucutureEnforcement.rawValue
        let klogPolicyPaths = getValidLogPolicyPaths()

        if secureConfigParameters == nil {
            validationError(key: secureConfigKey, explanation: "\(secureConfigKey) must be set with \(logPolicyKey) = \(klogPolicyPaths), \(metricsKey) = true, \(logFilteringKey) = true, and \(crashKey) = true", introducedInVersion: .two)
            return
        }
        
        var parameters:SecureConfigParameters
        do {
            let encoder = JSONEncoder()
            let data = try encoder.encode(secureConfigParameters)
            parameters = try SecureConfigParameters.decode(parametersJson: data, securityPolicy: self.policy)
        } catch {
            validationError(key: secureConfigKey, explanation: "Failed to parse \(secureConfigKey) parameters with error: \(error)", introducedInVersion: .two)
            return
        }
        
        for key in SecureConfigParameters.Keys.allCases {
            switch key {
            case .crashRedactionEnabled:
                if parameters.crashRedactionEnabled != true {
                    validationError(key: crashKey, subConfig: secureConfigKey, explanation: "\(crashKey) must be set to true", introducedInVersion: .two)
                }
            case .internalRequestOptionsAllowed:
                logger.debug("com.apple.tie.internalRequestOptionsAllowed is deprecated, please replace with com.apple.tie.allowNonProdExceptionOptions moving forward.")
                // allow any setting for now in carry
                break
            case .tie_allowNonProdExceptionOptions:
                // allow any setting for now in carry
                break
            case .logFilteringEnforced:
                if parameters.logFilteringEnforced != true {
                    validationError(key: logFilteringKey, subConfig: secureConfigKey, explanation: "\(logFilteringKey) must be set to true", introducedInVersion: .two)
                }
            case .logPolicyPath:
                if !isValidLogPolicyPath(path: parameters.logPolicyPath) {
                    validationError(key: logPolicyKey, subConfig: secureConfigKey, explanation: "\(logPolicyKey) must be set to \(klogPolicyPaths)", introducedInVersion: .two)
                }
            case .metricsFilteringEnforced:
                if parameters.metricsFilteringEnforced != true {
                    validationError(key: metricsKey, subConfig: secureConfigKey, explanation: "\(metricsKey) must be set to true", introducedInVersion: .two)
                }
            case .research_disableAppleInfrastrucutureEnforcement:
                if parameters.research_disableAppleInfrastrucutureEnforcement == true {
                    if Computer.isVM() {
                        logger.debug("Running in a VM, so \(appleInfrastrucutureEnforcementKey) is valid.")
                    } else {
                        validationError(key: appleInfrastrucutureEnforcementKey, subConfig: secureConfigKey, explanation: "\(appleInfrastrucutureEnforcementKey) must be set to false unless in a VM", introducedInVersion: .eight)
                    }
                }
            @unknown default:
                validationError(key: key.rawValue, subConfig: secureConfigKey, explanation: "Unknown key passed into validator, switch case needs to be updated", introducedInVersion: .one)
            }
        }
    }
}

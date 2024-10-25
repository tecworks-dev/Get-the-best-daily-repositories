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
//  DInitConfig+Apply.swift
//  darwin-init
//

import os
import System
import SystemConfiguration
import libnarrativecert
import CryptoKit

extension DInitConfig {
    private func print(logText: String) -> Bool {
        logger.log("darwin-init: \(logText, privacy: .public)")
        let banner = """
==== darwin-init ==============================================================
\(logText)
===============================================================================
"""
        do {
            try nothingOrErrno(retryOnInterrupt: true) {
                fputs(banner.cString(using: .utf8), stderr)
            }.get()
            return true
        } catch {
            logger.error("Failed to write to stderr: \(error.localizedDescription)")
            return false
        }
    }
    
    static internal func isValidConfig(cryptexConfig: [DInitCryptexConfig]) -> Bool {
        // validate that identifier field is unique
        
        // This will make a dictionary looks like:
        // {
        //      "com.apple.cryptex.foo": [{"url": "..."}, {"url": "..."}],
        //      "com.apple.cryptex.bar": [{"url": "..."}]
        // }
        //
        // Anything that has more than 1 element indicates there's duplicate identifier
        let crossRef = Dictionary(grouping: cryptexConfig, by: \.identifier)
        let duplicates = crossRef.filter { key, value in
            key != nil && value.count > 1
        }
        for duplicate in duplicates {
            if let identifier = duplicate.key {
                logger.error("Cryptex identifier \"\(identifier)\" is not unique")
            }
        }
        return duplicates.count == 0
    }

    func apply() async -> Self {
        logger.info("Applying darwin-init configuration...")

        var result = DInitConfig()
        
        // If we made it to apply then we passed validation
        // There is  nothing to apply for these keys, but set to avoid false failure
        result.configSecurityPolicy = configSecurityPolicy
        result.configSecurityPolicyVersion = configSecurityPolicyVersion
        result.applyTimeoutArgument = self.applyTimeoutArgument
        
        // Configure logging settings early so that client can see all info
        // in future logs for debugging
        if let logConfig = logConfig {
            result.logConfig = logConfig.apply()
                ? logConfig
                : nil
        }

#if os(macOS)
        if let userDefaultsDomain = UserDefaults(suiteName: "/Library/Application Support/CrashReporter/DiagnosticMessagesHistory.plist") {
            userDefaultsDomain.set(diagnosticsSubmissionEnabled ?? true, forKey:"AutoSubmit")
            result.diagnosticsSubmissionEnabled = diagnosticsSubmissionEnabled
        } else {
            result.diagnosticsSubmissionEnabled = nil
            logger.error("Failed to set diagnostic submission - could not create object for domain")
        }
#endif

        // Determine if we are booting into REM
        var bootingIntoREM = false
        if let purposeSetting = USRAfterSetup ?? Computer.defaultUSRAction() {
            do {
                let purpose = try DInitUSRPurpose.determineRebootPurpose(purposeSetting)
                bootingIntoREM = purpose == RB3_USERREBOOT_PURPOSE_ENTER_REM ||
                                 purpose == RB3_USERREBOOT_PURPOSE_ENTER_REM_DEVELOPMENT
            } catch {
                logger.error("Invalid setting in config \"\(DInitConfig.CodingKeys.USRAfterSetup)\": \"\(purposeSetting.rawValue)\"")
            }
        }

        // We need to set preference for the MTU of the uplink before
        // sending out the `firewall.installed` notification.
        if let mtu = networkUplinkMTU {
            logger.info("Setting uplink MTU to \(mtu)")
            result.networkUplinkMTU = Network.setUplinkMTU(mtu)
        }

        // Install firewall rules, if defined. Send a notification
        // if either installation of rules succeeded or there were
        // no rules defined.
        do {
            let firewallInstaller = FirewallInstaller()
            if let rules = firewallConfig?.rules {
                try firewallInstaller.installRules(rules)
                result.firewallConfig = firewallConfig
                logger.info("Installed firewall rules")
            }
            firewallInstaller.sendFirewallRulesInstalledEvent()
            logger.info("Sent notification com.apple.darwininit.firewall.installed")
        } catch {
            logger.error("Unable to install firewall rules: \(error)")
        }

        // Pre-init commands must run before everything else.
        result.preInitCritical = preInitCritical
        let preInitCritical = preInitCritical ?? false
        if let preInitCommands = preInitCommands {
            result.preInitCommands = []
            for (i, command) in preInitCommands.enumerated() {
                logger.info("Executing pre-init command \(i+1): \(command)")
                if Subprocess.run(shell: "/bin/bash", command: command) {
                    result.preInitCommands?.append(command)
                } else {
                    logger.error("pre-init command \(i+1) failed.")
                    guard !preInitCritical else {
                        logger.error("Critical pre-init command failed: '\(command)'")
                        return result
                    }
                }
            }
        }

        if let collectPerfData = collectPerfData {
            result.collectPerfData = collectPerfData
                ? PerformanceData.writeStats(to: "/var/tmp/bootperf.pdj")
                : nil
        }

        if let tailspinConfig = tailspinConfig {
            result.tailspinConfig = tailspinConfig.apply() ? tailspinConfig : nil
        }

        // Set deprecated fields if provided. Newer fields will replace these
        // fields if present.
        if let legacyComputerName = legacyComputerName {
            do {
                try await Computer.set(computerName: legacyComputerName)
                try await Computer.set(localHostName: legacyComputerName)
                result.legacyComputerName = legacyComputerName
            } catch {
                logger.error("Failed to set legacyComputerName: \(error)")
            }
        }

        if let legacyFQDN = legacyFQDN {
            do {
                try await Computer.set(computerName: legacyFQDN)
                result.legacyFQDN = legacyFQDN
            } catch {
                logger.error("Failed to set legacyFQDN: \(error)")
            }
        }

        if let legacyHostName = legacyHostName {
            do {
                try await Computer.set(localHostName: legacyHostName)
                result.legacyHostName = legacyHostName
            } catch {
                logger.error("Failed to set legacyHostName: \(error)")
            }
        }

        if let computerName = computerName {
            do {
                try await Computer.set(computerName: computerName)
                result.computerName = computerName
            } catch {
                logger.error("Failed to set computerName: \(error)")
            }
        }

        if let hostName = hostName {
            do {
                try await Computer.set(hostName: hostName)
                result.hostName = hostName
            } catch {
                logger.error("Failed to set hostName: \(error)")
            }
        }

        if let localHostName = localHostName {
            do {
                try await Computer.set(localHostName: localHostName)
                result.localHostName = localHostName
            } catch {
                logger.error("Failed to set localHostName: \(error)")
            }
        }
        
        if let usageLabel = usageLabel {
            if UsageLabel.setUsageLabel(to: usageLabel) && Computer.setAutomatedDeviceGroup(to: usageLabel) {
                result.usageLabel = usageLabel
            } else {
                logger.error("Failed to set \(kUsageLabelKey, privacy: .public).")
            }
        }

        // install roots if asked. This is done early so that other steps can use them.
        if let caRoots = caRoots {
            result.caRoots = caRoots.apply()
                ? caRoots
                : nil
        }

        // NOTE: Apply any user configuration before enabling SSH so that any provided credentials
        // will be usable as soon as sshd is listening for connections.
        if let userConfig = userConfig {
            result.userConfig = userConfig.apply()
                ? userConfig
                : nil
        }

        do {
            switch enableSSHPasswordAuthentication {
            case .some(true):
                try kDInitSSHConfig.save(kDInitSSHConfigPWAuthEnabled, append: true)
            case .some(false):
                try kDInitSSHConfig.save(kDInitSSHConfigPWAuthDisabled, append: true)
            case nil:
                try kDInitSSHConfig.save(kDInitSSHConfigEmpty, append: true)
            }
            result.enableSSHPasswordAuthentication = self.enableSSHPasswordAuthentication
        } catch {
            logger.error("Failed to configure SSH password authentication: \(error)")
            result.enableSSHPasswordAuthentication = nil
        }

        if let enableSSH = enableSSH {
#if os(macOS)
            let sshPlist = "/System/Library/LaunchDaemons/ssh.plist"
            result.enableSSH = enableSSH
            ? Subprocess.run(shell: nil, command: "/bin/launchctl load -F -w \(sshPlist)")
                : false
#else
            let sshPlist = "/AppleInternal/Library/LaunchDaemons/com.apple.internal.darwin.ssh.plist"
            // launchctl is mastered out of customer embedded darwinOS
            // So, do nothing but report successful application in the result config
            if !os_variant_has_internal_content(Logger.subsystem) {
                result.enableSSH = enableSSH
            } else {
                result.enableSSH = enableSSH
                ? Subprocess.run(shell: nil, command: "/bin/launchctl load -F -w \(sshPlist)")
                : false
            }
#endif
        }

        if let issueDCRT = issueDCRT {
            do {
                try await DCRT.issue()
                result.issueDCRT = issueDCRT
            } catch let error as NSError {
                logger.error("Failed to issue DCRT: \(error, privacy: .public)")
                logger.error("DCRT root error: \(error.rootError(), privacy: .public)")
            }
        }

        if let installConfig = installConfig {
            result.installConfig = installConfig.apply()
                ? installConfig
                : nil
        }

        if let preferencesConfig = preferencesConfig {
            let applied = preferencesConfig.compactMap {
                $0.apply() ? $0 : nil
            }
            CFPreferences.flushCaches()
            let verified = applied.compactMap {
                $0.verify() ? $0 : nil
            }
            result.preferencesConfig = verified
        }
        
        /* Only apply network config if firewall was installed and
           this is not a compute node */
        if result.firewallConfig == firewallConfig && !Computer.isComputeNode() {
            if let networkConfig = networkConfig {
                let applied = networkConfig.compactMap {
                    $0.apply() ? $0 : nil
                }
                let verified = applied.compactMap {
                    $0.verify() ? $0 : nil
                }
                result.networkConfig = verified
            }
        }

        if let narrativeIdentitiesConfig = narrativeIdentitiesConfig {
            result.narrativeIdentitiesConfig = narrativeIdentitiesConfig.compactMap {
                $0.apply() ? $0 : nil
            }
        }
        
        if let diavloConfig {
            result.diavloConfig = await diavloConfig.apply()
        }

        // Set the bandwidth limit on the Mellanox interface before downloading cryptexes
        if let bandwidthLimit = bandwidthLimit {
            result.bandwidthLimit = Network.setUplinkBandwidthLimit(bandwidthLimit: bandwidthLimit) ? bandwidthLimit : nil
        }

        // If a bandwidth limit was specified, only download if it was set successfully
        if (bandwidthLimit != nil && result.bandwidthLimit != nil) {
            logger.error("Failed to set bandwidth limit for Mellanox interface. Will NOT download cryptexes.")
        }
        
        if (bandwidthLimit == nil || (bandwidthLimit != nil && result.bandwidthLimit != nil)), let cryptexConfig = cryptexConfig, DInitConfig.isValidConfig(cryptexConfig: cryptexConfig) {
            // Cache entries that are not for cryptexes in the input configs are removed to maintain consistency with ephemeral data mode.
            // This can be optionally disabled, but only for internal users (on dev-fused hardware).
            let removeCacheEntriesNotInInputSet: Bool
            if os_variant_allows_internal_security_policies(Logger.subsystem) {
                removeCacheEntriesNotInInputSet = retainPreviouslyCachedCryptexesUnsafely != true
            } else {
                if retainPreviouslyCachedCryptexesUnsafely == true {
                    result.retainPreviouslyCachedCryptexesUnsafely = nil
                }

                removeCacheEntriesNotInInputSet = true
            }

#if os(watchOS)
            let defaultCryptexCacheMaxTotalSize: Int = Int.max / 2
#else
            let defaultCryptexCacheMaxTotalSize: Int = (256 * 1024 * 1024 * 1024)
#endif

            let cryptexCacheMaxTotalSize: Int = max(0, self.cryptexCacheMaxTotalSize ?? defaultCryptexCacheMaxTotalSize)
            if self.cryptexCacheMaxTotalSize != nil {
                result.cryptexCacheMaxTotalSize = cryptexCacheMaxTotalSize
            }

            let extractedCryptexes: [DInitCryptexConfig.ExtractedCryptex]
            if let cryptexCache = CryptexCache(at: CryptexCache.defaultCacheDirectoryPath, maxTotalSize: cryptexCacheMaxTotalSize) {
                do {
                    extractedCryptexes = try await cryptexCache.fetch(configs: cryptexConfig, removeCacheEntriesNotInInputSet: removeCacheEntriesNotInInputSet) { config, result -> DInitCryptexConfig.ExtractedCryptex? in
                        switch result {
                        case .nonCacheable:
                            return await config.download()?.extract()
                        case .found(let extracted):
                            return extracted
                        case .cachingFailure(let downloadedCryptex):
                            return await downloadedCryptex.extract()
                        }
                    }
                } catch {
                    logger.error("CryptexCache.fetch() failed: \(error)")
                    // If we try to retry something here, we may apply something twice in the case of a partial failure.
                    result.cryptexConfig = nil
                    extractedCryptexes = []
                }
            } else {
                logger.error("Failed to initialize CryptexCache; proceeding without caching")
                extractedCryptexes = await cryptexConfig.asyncSerialCompactMap { await $0.download()?.extract() }
            }

            let personalizedCryptexes = extractedCryptexes.compactMap { $0.personalize(diavloServerURL: diavloConfig?.serverURL) }

            let applied = personalizedCryptexes.compactMap { $0.install(limitLoadToREM: bootingIntoREM) ? $0.config : nil }
            result.cryptexConfig = applied
        }
        
        // Reset the bandwidth limit on the Mellanox interface after cryptex downloads
        // If we fail to reset the bandwidth limit, we consider this an apply failure
        if bandwidthLimit != nil && result.bandwidthLimit != nil {
            result.bandwidthLimit = Network.unsetUplinkBandwidthLimit() ? bandwidthLimit : nil
        }

        if bootingIntoREM || lockCryptexes ?? false {
            do {
                try Computer.lockCryptexes()
                result.lockCryptexes = lockCryptexes
            } catch {
                logger.error("Locking sealed software hash registers failed: \(error)")
            }
        }

        // If not booting into rem and cryptex lockdown is disabled, convey this in applied config
        if !bootingIntoREM && lockCryptexes == false {
            result.lockCryptexes = lockCryptexes
        }

        if let packageConfig = packageConfig {
            result.packageConfig = await packageConfig.asyncSerialCompactMap {
                await $0.apply()
            }
        }

        if let logText = logText {
            result.logText = print(logText: logText)
                ? logText
                : nil
        }

        result.rebootAfterSetup = rebootAfterSetup
        result.USRAfterSetup = USRAfterSetup
        result.resultConfig = resultConfig

        return result
    }
}

extension DInitCryptexConfig {

    // FIXME: Do we need to mitigate any security concerns from this?
    /// n.b. the preflight, darwinup and postflight are executed at a higher privilege
    /// than normal root processes because of the
    /// `com.apple.private.security.storage-exempt.heritable` entitlement.

    func download(to destinationDirectory: FilePath? = nil) async -> DownloadedCryptex? {
        logger.info("Downloading cryptex...")

        // If cryptex url has knox:// scheme, handle this special case
        if url.scheme == kKnoxURLScheme {
            // Download the raw, encrypted cryptex from Knox
            guard let encrypted = await KnoxClientWrapper.downloadRaw(at: url, to: destinationDirectory, dawToken: dawToken, wgUsername: wgUsername, wgToken: wgToken, altCDN: alternateCDNHost, background: backgroundTrafficClass, retries: networkRetryCount) else {
                return nil
            }

            return DownloadedCryptex(config: self, path: encrypted)
        }

        guard let compressed = await Network.downloadItem(at: url, to: destinationDirectory, attempts: 5) else {
            return nil
        }

        if let size = size {
            do {
                try compressed.sizeEquals(expectedSize: size)
            } catch {
                logger.error("Downloaded cryptex failed size validation: \(error.localizedDescription)")
                return nil
            }
        }
        if let sha256 = sha256 {
            do {
                try compressed.sha256Equals(expectedSHA256: sha256)
            } catch {
                logger.error("Downloaded cryptex failed sha256 validation: \(error.localizedDescription)")
                return nil
            }
        }

        return DownloadedCryptex(config: self, path: compressed)
    }

    struct DownloadedCryptex {
        let config: DInitCryptexConfig
        let path: FilePath

        func extract() async -> ExtractedCryptex? {
            guard let extracted = await LibCryptex.extractCryptex(
                at: path,
                url: config.url,
                dawToken: config.dawToken,
                wgUsername: config.wgUsername,
                wgToken: config.wgToken,
                altCDN: config.alternateCDNHost,
                retries: config.networkRetryCount,
                aeaDecryptionParams: config.aeaDecryptionParams) else {
                return nil
            }

            return ExtractedCryptex(config: config, path: extracted)
        }
    }

    struct ExtractedCryptex {
        let config: DInitCryptexConfig
        let path: FilePath

        func personalize(diavloServerURL: String?) -> PersonalizedCryptex? {
            let serverURL = (config.auth == .diavlo) ? diavloServerURL : nil

            guard let personalized = LibCryptex.personalizeCryptex(
                at: path,
                withVariant: config.variant,
                usingAuthorizationService: config.auth,
                locatedAt: serverURL,
                usingAppleConnect: config.appleConnect ?? false) else {
                return nil
            }

            return PersonalizedCryptex(config: config, path: personalized, serverURL: serverURL)
        }
    }

    struct PersonalizedCryptex {
        let config: DInitCryptexConfig
        let path: FilePath
        let serverURL: String?

        func install(limitLoadToREM: Bool) -> Bool {
            return LibCryptex.installCryptex(
                at: path,
                withVariant: config.variant,
                usingAuthorizationService: config.auth,
                locatedAt: serverURL,
                limitLoadToREM: limitLoadToREM)
        }
    }

    struct ResolvedComponents {
        var archiveID: Data
        var key: SymmetricKey
    }
    static func resolveDecryptionComponents(url: URL?, dawToken: String?, wgUsername: String?, wgToken: String?, altCDN: String?, retries: UInt?, aeaDecryptionParams: DInitAEADecryptionParams?) async -> ResolvedComponents? {
        
        if aeaDecryptionParams != nil && url?.scheme == kKnoxURLScheme {
            logger.warning("Pre-determined decryption components already provided for a knox:// url. Will use pre-determined components rather than fetching from Knox.")
        }
        
        if let aeaDecryptionParams {
            logger.info("Pre-determined decryption components provided")
            guard let key = aeaDecryptionParams.getDecryptionKey() else {
                return nil
            }
            guard let archiveIdentifierData = aeaDecryptionParams.getArchiveID() else {
                return nil
            }
            return ResolvedComponents(archiveID: archiveIdentifierData, key: key)
        }
        
        if url?.scheme == kKnoxURLScheme {
            logger.log("Attempting to fetch decryption key for cryptex from Knox.")
            
            guard let decryptionComponents = await KnoxClientWrapper.getDecryptionComponents(for: url!, dawToken: dawToken, wgUsername: wgUsername, wgToken: wgToken, altCDN: altCDN, retries: retries),
                  let key = KnoxClientWrapper.getDecryptionKey(from: decryptionComponents) else {
                return nil
            }
            logger.debug("decryption-components.digest-algorithm: \(decryptionComponents.digestAlgorithm)")
            if !decryptionComponents.digestAlgorithm.starts(with: "APPLE-ARCHIVE-IDENTIFIER") {
                logger.warning("Decryption components digest doesn't appear to be an Apple Archive identifier!")
            }
            guard let archiveIdentifierBytes = decryptionComponents.digest.hexadecimalASCIIBytes, !archiveIdentifierBytes.isEmpty else {
                logger.error("Failed to convert archive identifier from String to Data")
                return nil
            }
            return ResolvedComponents(archiveID: Data(archiveIdentifierBytes), key: key)
        }
        return nil
    }
}

extension DInitTailSpinConfig {
    func apply() -> Bool {
        logger.info("Applying tailspin configuration...")
        if (shim_check_tailspin()) {
            var success = false
            guard var tailspingconfig = shim_tailspin_config_create_with_default_config() else {
                logger.error("Failed to generate tailspin config.")
                return false;
            }

            self.processConfig(tailspin_config:&tailspingconfig)
            success = shim_tailspin_config_apply_sync(tailspingconfig)
            defer {shim_tailspin_config_free(tailspingconfig)}

            if !success {
                logger.error("Failed to apply tailspin configuration.")
            }
            return success
        } else {
            logger.error("Tailspin doesn't exist in on this OS, so we cannot handle tailspin configuration.")
            return false;
        }
    }
}

extension DInitDiavloConfig {
    func fetchCertsFromServer() async throws -> Bool {
        logger.info("Applying diavlo configuration...")

        let rootCertificates: DiavloCertList
        if let _rootCertificate = self.rootCertificate {
            logger.debug("Using diavlo root certificate provided in darwin-init config")
            let diavloCert = DiavloCert(cert: _rootCertificate.rawValue)
            rootCertificates = DiavloCertList(certificates: [diavloCert])
        } else {
            logger.debug("Fetching diavlo root certificate from server at \(serverURL)")
            rootCertificates = try await DiavloClient.fetchRootCertificate(server: serverURL)
        }

        return rootCertificates
            .certificates
            .map {
                LibCryptex.trust(
                    rootCertificate: $0.cert,
                    usingAppleConnect: appleConnect ?? false,
                    signingURL: serverURL)
            }
            .reduce(false) { partialResult, incrementalTrustResult in
                partialResult || incrementalTrustResult
            }
    }

    func apply() async -> Self? {
        var success = false;
        do {
            success = try await self.fetchCertsFromServer()
        } catch {
            logger.error("Application of diavlo config error \(error).")
        }

        return success ? self : nil
    }

}

extension DInitInstallConfig {
    func apply() -> Bool {
        logger.info("Applying install configuration...")

        if let waitForVolume = waitForVolume {
            guard FilePath(waitForVolume).exists(withTimeout: 20) else {
                logger.error("Wait for volume \(waitForVolume) failed")
                return false
            }
        }

        if let preflight = preflight {
            let shell = self.preflightShell ?? "/bin/bash"
            guard Subprocess.run(shell: shell, command: preflight) else {
                logger.error("Preflight failed")
                return false
            }
        }

        if let root = root {
            let command = "/usr/bin/darwinup install \(root)"
            guard Subprocess.run(shell: nil, command: command) else {
                logger.error("Root installed failed")
                return false
            }
        }

        if let postflight = postflight {
            let shell = self.postflightShell ?? "/bin/bash"
            guard Subprocess.run(shell: shell, command: postflight) else {
                logger.error("Postflight failed")
                return false
            }
        }

        return true
    }
}

extension DInitNetworkConfig {
    func apply() -> Bool {
        logger.info("Applying network configuration...")

        return NetworkConfig.setConfig(retryLimit: 5, config: value.dictionaryValue, interface: interface)
    }
 
    func verify() -> Bool {
        CFEqual(value.dictionaryValue ?? kCFNull, NetworkConfig.getConfig(interface: interface, config: value.dictionaryValue) ?? kCFNull)
    }
}

extension DInitPreferencesConfig {
    private var domain: CFPreferences.Domain {
        CFPreferences.Domain(
            applicationId: applicationId as CFString? ?? kCFPreferencesAnyApplication,
            userName: userName as CFString? ?? kCFPreferencesAnyUser,
            hostName: hostName as CFString? ?? kCFPreferencesCurrentHost)
    }

    func apply() -> Bool {
        logger.info("Applying preference configuration...")

        CFPreferences.set(value: value.propertyListValue, for: key, in: domain)
        return CFPreferences.synchronize(domain: domain)
    }

    func verify() -> Bool {
        CFEqual(
            value.propertyListValue ?? kCFNull,
            CFPreferences.getValue(for: key, in: domain) ?? kCFNull)
    }
}

extension DInitNarrativeIdentitiesConfig {

    func apply() -> Bool {
        logger.info("Applying narrative identities configuration...")

        do {
            let components = identity.components(separatedBy: "-")
            guard components.count == 2,
                  isNarrativeCertSupported(domain: components[0], identityType: components[1]),
                  let narrativeDomain = NarrativeDomain(rawValue: components[0]),
                  let narrativeIdentityType = NarrativeIdentityType(rawValue: components[1]) else {
                    logger.error("Narrative Identity \(identity) is not supported. Supported Identities are \(formatSupportedNarrativeIdentities())")
                return false
            }
            
            let client = NarrativeXPCClient()
            try client.ConfigureIdentity(narrativeDomain: narrativeDomain, narrativeIdentityType: narrativeIdentityType)
        } catch {
            logger.error("Error when configuring identity for \(identity): \(error.localizedDescription)")
            return false
        }
        
        return true
    }
}


extension DInitPackageConfig {
    func apply() async -> DInitPackageConfig? {
        logger.info("Applying package configuration...")

#if os(macOS)
        guard let packagePath = await Network.downloadItem(at: url) else {
            return nil
        }

        let installer = PackageInstaller(url: URL(fileURLWithPath: "\(packagePath)"))
        return installer.install() ? self : nil
#else
        logger.fault("Package Install has no effect because it is not available on this platform")
        return nil
#endif
    }
}

extension DInitCARoots {
    func apply() -> Bool {
        if appleCorpRoot == true {
            guard CorporateRoot.apply() else {
                return false
            }
        }
        return true
    }
}

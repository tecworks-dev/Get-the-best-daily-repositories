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

//  Copyright © 2023 Apple Inc. All rights reserved.

import CFPreferenceCoder
import CloudBoardCommon
import Foundation
import NIOCore

extension CFPreferences {
    static var cbAttestationPreferencesDomain: String {
        "com.apple.cloudos.cb_attestationd"
    }
}

struct CloudBoardAttestationDConfiguration: Codable, Hashable {
    enum CodingKeys: String, CodingKey {
        case _useInMemoryKey = "UseInMemoryKey"
        case _keyLifetimeMinutes = "KeyLifetimeMinutes"
        case _keyExpiryGracePeriodSeconds = "KeyExpiryGracePeriodSeconds"
        case _cloudAttestation = "CloudAttestation"
    }

    private var _useInMemoryKey: Bool?
    private var _keyLifetimeMinutes: Int?
    private var _keyExpiryGracePeriodSeconds: Int?
    private var _cloudAttestation: CloudAttestationConfiguration?

    // If true, an in-memory key is used as node key instead of a SEP-backed key. This automatically disables the use of
    // CloudAttestation which require SEP-backed keys. Note that regular clients enforce SEP-backed attested keys and
    // with that cannot talk to nodes with this value set to true.
    var useInMemoryKey: Bool {
        get { self._useInMemoryKey ?? false }
        set { self._useInMemoryKey = newValue }
    }

    /// Lifetime of node keys. Note that in production, the configured key lifetime plus the configured grace period
    /// together is enforced to be below 48 hours.
    var keyLifetime: TimeAmount {
        self._keyLifetimeMinutes.map { .minutes(Int64($0)) } ?? .hours(24)
    }

    /// Grace period in which keys are usable beyond their advertised expiry/lifetime
    var keyExpiryGracePeriod: TimeAmount {
        self._keyExpiryGracePeriodSeconds.map { .seconds(Int64($0)) } ?? .minutes(5)
    }

    /// CloudAttestation-related configuration. See individual fields for documentation.
    var cloudAttestation: CloudAttestationConfiguration {
        get { return self._cloudAttestation ?? CloudAttestationConfiguration() }
        set { self._cloudAttestation = newValue }
    }

    init(useInMemoryKey: Bool, cloudAttestation: CloudAttestationConfiguration) {
        self._useInMemoryKey = useInMemoryKey
        self._cloudAttestation = cloudAttestation
    }

    static func fromFile(
        path: String,
        secureConfigLoader: SecureConfigLoader
    ) throws -> CloudBoardAttestationDConfiguration {
        var config: CloudBoardAttestationDConfiguration
        do {
            CloudBoardAttestationDaemon.logger.info("Loading configuration from file \(path, privacy: .public)")
            let fileContents = try Data(contentsOf: URL(filePath: path))
            let decoder = PropertyListDecoder()
            config = try decoder.decode(CloudBoardAttestationDConfiguration.self, from: fileContents)
        } catch {
            CloudBoardAttestationDaemon.logger.error(
                "Unable to load config from file: \(String(unredacted: error), privacy: .public)"
            )
            throw error
        }

        let secureConfig: SecureConfig
        do {
            CloudBoardAttestationDaemon.logger.info("Loading secure config from SecureConfigDB")
            secureConfig = try secureConfigLoader.load()
        } catch {
            CloudBoardAttestationDaemon.logger
                .error("Error loading secure config: \(String(unredacted: error), privacy: .public)")
            throw error
        }

        if secureConfig.forceSecurityConfigurationOn {
            config.enforceSecurityConfig()
        }
        return config
    }

    static func fromPreferences(secureConfigLoader: SecureConfigLoader = .real) throws
    -> CloudBoardAttestationDConfiguration {
        CloudBoardAttestationDaemon.logger
            .info(
                "Loading configuration from preferences \(CFPreferences.cbAttestationPreferencesDomain, privacy: .public)"
            )
        let preferences = CFPreferences(domain: CFPreferences.cbAttestationPreferencesDomain)
        do {
            return try .fromPreferences(preferences, secureConfigLoader: secureConfigLoader)
        } catch {
            CloudBoardAttestationDaemon.logger.error(
                "Error loading configuration from preferences: \(String(unredacted: error), privacy: .public)"
            )
            throw error
        }
    }

    mutating func enforceSecurityConfig() {
        CloudBoardAttestationDaemon.logger.debug("Enforcing security config")
        if self.useInMemoryKey {
            CloudBoardAttestationDaemon.logger.error("Overriding configuration with new value UseInMemoryKey=false")
            self.useInMemoryKey = false
        }

        if !self.cloudAttestation.enabled {
            CloudBoardAttestationDaemon.logger
                .error("Overriding configuration with new value CloudAttestation.Enabled=true")
            self.cloudAttestation.enabled = true
        }
    }

    static func fromPreferences(
        _ preferences: CFPreferences,
        secureConfigLoader: SecureConfigLoader
    ) throws -> CloudBoardAttestationDConfiguration {
        let decoder = CFPreferenceDecoder()
        var configuration = try decoder.decode(CloudBoardAttestationDConfiguration.self, from: preferences)

        let secureConfig: SecureConfig
        do {
            CloudBoardAttestationDaemon.logger.info("Loading secure config from SecureConfigDB")
            secureConfig = try secureConfigLoader.load()
        } catch {
            CloudBoardAttestationDaemon.logger
                .error("Error loading secure config: \(String(unredacted: error), privacy: .public)")
            throw error
        }

        if secureConfig.forceSecurityConfigurationOn {
            configuration.enforceSecurityConfig()
        }

        return configuration
    }
}

extension CloudBoardAttestationDConfiguration {
    struct CloudAttestationConfiguration: Codable, Hashable {
        enum CodingKeys: String, CodingKey {
            // Determines whether we provide a real CloudAttestation-provided attestation bundle or a fake attestation
            // bundle that just contains the public key
            case _enabled = "Enabled"
            // Enables inclusion of a transparency inclusion proof in the attestation bundle
            case _includeTransparencyLogInclusionProof = "IncludeTransparencyLogInclusionProof"
            case _attestationRetryConfiguration = "AttestationRetryConfiguration"
        }

        private var _enabled: Bool?
        private var _includeTransparencyLogInclusionProof: Bool?
        private var _attestationRetryConfiguration: RetryConfiguration?

        /// If true, CloudAttestation is used to attest node keys and create attestation bundles. Otherwise, a fake
        /// attestation bundle only containing the public node key in OHTTP key configuration encoding is generated.
        var enabled: Bool {
            get { self._enabled ?? true }
            set { self._enabled = newValue }
        }

        /// If true, an transparency log inclusion proof is fetched from the Transparency Service and included in the
        /// attestation bundle
        var includeTransparencyLogInclusionProof: Bool {
            get { self._includeTransparencyLogInclusionProof ?? false }
            set { self._includeTransparencyLogInclusionProof = newValue }
        }

        /// Configuration of node key attestation generation retries. See individual fields for documentation.
        var attestationRetryConfiguration: RetryConfiguration {
            self._attestationRetryConfiguration ?? .init()
        }

        init(
            enabled: Bool? = true,
            includeTransparencyLogInclusionProof: Bool = false,
            _attestationRetryConfiguration: RetryConfiguration? = nil
        ) {
            self._enabled = enabled
            self._includeTransparencyLogInclusionProof = includeTransparencyLogInclusionProof
            self._attestationRetryConfiguration = _attestationRetryConfiguration
        }
    }
}

extension CloudBoardAttestationDConfiguration {
    struct RetryConfiguration: Codable, Hashable {
        enum CodingKeys: String, CodingKey {
            case _initialDelaySeconds = "InitialDelaySeconds"
            case _multiplier = "Multiplier"
            case _maxDelaySeconds = "MaxDelaySeconds"
            case jitter = "Jitter"
            case _perRetryTimeoutSeconds = "PerRetryTimeoutSeconds"
            case _timeoutSeconds = "TimeoutSeconds"
        }

        private var _initialDelaySeconds: Int64?
        private var _multiplier: Double?
        private var _maxDelaySeconds: Int64?
        private var _perRetryTimeoutSeconds: Int64?
        private var _timeoutSeconds: Int64?

        /// Initial delay between retries
        var initialDelay: Duration { self._initialDelaySeconds.map { .seconds($0) } ?? .seconds(1) }
        /// Multiplier for exponential backoff between retries, starting from initial delay
        var multiplier: Double { self._multiplier ?? 1.6 }
        /// Maximum delay between retries
        var maxDelay: Duration { self._maxDelaySeconds.map { .seconds($0) } ?? .minutes(5) }
        /// Jitter in percent
        var jitter: Double?
        // Per-retry timeout
        var perRetryTimeout: Duration? { self._perRetryTimeoutSeconds.map { .seconds($0) } }
        /// Overall timeout after which no new retry is attempted
        var timeout: Duration? {
            get { self._timeoutSeconds.map { .seconds($0) } }
            set { self._timeoutSeconds = newValue?.components.seconds }
        }

        init(
            initialDelay: Duration? = nil,
            multiplier: Double? = nil,
            maxDelay: Duration? = nil,
            jitter: Double? = nil,
            perRetryTimeout: Duration? = nil,
            timeout: Duration? = nil
        ) {
            self._initialDelaySeconds = initialDelay?.components.seconds
            self._multiplier = multiplier
            self._maxDelaySeconds = maxDelay?.components.seconds
            self.jitter = jitter
            self._perRetryTimeoutSeconds = perRetryTimeout?.components.seconds
            self._timeoutSeconds = timeout?.components.seconds
        }
    }
}

extension Duration {
    static func minutes(_ minutes: Int) -> Duration {
        .init(secondsComponent: Int64(minutes * 60), attosecondsComponent: 0)
    }
}

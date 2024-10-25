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

extension CFPreferences {
    /// The domain used for the configuration daemon's preferences.
    static var cbConfigPreferencesDomain: String {
        "com.apple.cloudos.cb_configurationd"
    }
}

/// The static configuration of the configuration daemon.
struct ConfigurationDConfiguration: Decodable, Hashable {
    enum CodingKeys: String, CodingKey {
        case _fetcherSchedule = "FetcherSchedule"
        case _configurationService = "ConfigurationService"
        case _releaseInfo = "ReleaseInfo"
        case _launchDelay = "LaunchDelay"
    }

    struct FetcherSchedule: Decodable, Hashable {
        private var _tickInterval: Double?
        private var _maximumToleranceRatio: Double?
        private var _minimumInterTickDelayRatio: Double?

        init(
            tickInterval: Double?,
            maximumToleranceRatio: Double?,
            minimumInterTickDelayRatio: Double?
        ) {
            self._tickInterval = tickInterval
            self._maximumToleranceRatio = maximumToleranceRatio
            self._minimumInterTickDelayRatio = minimumInterTickDelayRatio
        }

        static var `default`: Self {
            .init(
                tickInterval: nil,
                maximumToleranceRatio: nil,
                minimumInterTickDelayRatio: nil
            )
        }

        enum CodingKeys: String, CodingKey {
            case _tickInterval = "TickInterval"
            case _maximumToleranceRatio = "MaximumToleranceRatio"
            case _minimumInterTickDelayRatio = "MinimumInterTickDelayRatio"
        }

        static var defaultTickInterval: Double = 60.0
        static var defaultMaximumToleranceRatio: Double = 0.25
        static var defaultMinimumInterTickDelayRatio: Double = 0.10

        var tickInterval: Double {
            self._tickInterval ?? Self.defaultTickInterval
        }

        var maximumToleranceRatio: Double {
            self._maximumToleranceRatio ?? Self.defaultMaximumToleranceRatio
        }

        var minimumInterTickDelayRatio: Double {
            self._minimumInterTickDelayRatio ?? Self.defaultMinimumInterTickDelayRatio
        }
    }

    struct ConfigurationService: Decodable, Hashable {
        private var _metadataURL: URL
        private var _storageURL: URL

        private var _attemptCount: Int?
        private var _retryDelay: TimeInterval?
        private var _httpRequestTimeout: TimeInterval?
        private var _maximumConfigurationPackageSize: Int?

        init(
            metadataURL: URL,
            storageURL: URL,
            attemptCount: Int? = nil,
            retryDelay: TimeInterval? = nil,
            httpRequestTimeout: TimeInterval? = nil,
            maximumConfigurationPackageSize: Int? = nil
        ) {
            self._metadataURL = metadataURL
            self._storageURL = storageURL
            self._attemptCount = attemptCount
            self._retryDelay = retryDelay
            self._httpRequestTimeout = httpRequestTimeout
            self._maximumConfigurationPackageSize = maximumConfigurationPackageSize
        }

        enum CodingKeys: String, CodingKey {
            case _metadataURL = "MetadataURL"
            case _storageURL = "StorageURL"
            case _attemptCount = "AttemptCount"
            case _retryDelay = "RetryDelay"
            case _httpRequestTimeout = "HTTPRequestTimeout"
            case _maximumConfigurationPackageSize = "MaximumConfigurationPackageSize"
        }

        var metadataURL: URL {
            self._metadataURL
        }

        var storageURL: URL {
            self._storageURL
        }

        var attemptCount: Int {
            self._attemptCount ?? 3
        }

        var retryDelay: TimeInterval {
            self._retryDelay ?? 5.0
        }

        var httpRequestTimeout: TimeInterval {
            self._httpRequestTimeout ?? 30.0
        }

        var maximumConfigurationPackageSize: Int {
            self._maximumConfigurationPackageSize ?? 5 * 1024 * 1024
        }
    }

    struct ReleaseInfo: Decodable, Hashable {
        private var _project: String
        private var _environment: String
        private var _release: String

        init(
            project: String,
            environment: String,
            release: String
        ) {
            self._project = project
            self._environment = environment
            self._release = release
        }

        enum CodingKeys: String, CodingKey {
            case _project = "Project"
            case _environment = "Environment"
            case _release = "Release"
        }

        var project: String {
            self._project
        }

        var environment: String {
            self._environment
        }

        var release: String {
            self._release
        }
    }

    private var _fetcherSchedule: FetcherSchedule?
    private var _configurationService: ConfigurationService?
    private var _releaseInfo: ReleaseInfo?
    private var _launchDelay: Int?

    init(
        fetcherSchedule: FetcherSchedule? = nil,
        configurationService: ConfigurationService? = nil,
        releaseInfo: ReleaseInfo? = nil,
        launchDelay: Int? = nil
    ) {
        self._fetcherSchedule = fetcherSchedule
        self._configurationService = configurationService
        self._releaseInfo = releaseInfo
        self._launchDelay = launchDelay
    }

    /// The fetcher used to compute the time between the individual pulls of config.
    var fetcherSchedule: FetcherSchedule {
        self._fetcherSchedule ?? .default
    }

    /// The information required to connect to and maintain a connection with the configuration service.
    var configurationService: ConfigurationService? {
        self._configurationService
    }

    /// The information required to fetch the revision and configuration package from the configuration service.
    var releaseInfo: ReleaseInfo? {
        self._releaseInfo
    }

    /// The delay in seconds the daemon should apply so that the logging daemon has a chance to come up first
    /// and we avoid losing logs.
    var launchDelay: Int {
        self._launchDelay ?? 15
    }

    internal enum ConfigurationDConfigurationError: Error {
        case disallowedURLScheme
    }

    static func fromFile(path: String, secureConfigLoader: SecureConfigLoader) throws -> ConfigurationDConfiguration {
        var config: ConfigurationDConfiguration
        do {
            ConfigurationDaemon.logger.notice("Loading configuration from file \(path, privacy: .public)")
            let fileContents = try Data(contentsOf: URL(filePath: path))
            let decoder = PropertyListDecoder()
            config = try decoder.decode(ConfigurationDConfiguration.self, from: fileContents)
        } catch {
            ConfigurationDaemon.logger
                .error("Unable to load config from file: \(String(unredacted: error), privacy: .public)")
            throw error
        }

        let secureConfig: SecureConfig
        do {
            ConfigurationDaemon.logger.info("Loading secure config from SecureConfigDB")
            secureConfig = try secureConfigLoader.load()
        } catch {
            ConfigurationDaemon.logger.error(
                "Error loading secure config: \(String(unredacted: error), privacy: .public)"
            )
            throw error
        }

        if secureConfig.shouldEnforceAppleInfrastructureSecurityConfig {
            try config.enforceAppleInfrastructureSecurityConfig()
        }
        return config
    }

    static func fromPreferences(secureConfigLoader: SecureConfigLoader = .real) throws -> ConfigurationDConfiguration {
        ConfigurationDaemon.logger
            .info("Loading configuration from preferences \(CFPreferences.cbConfigPreferencesDomain, privacy: .public)")
        let preferences = CFPreferences(domain: CFPreferences.cbConfigPreferencesDomain)
        do {
            return try .fromPreferences(preferences, secureConfigLoader: secureConfigLoader)
        } catch {
            ConfigurationDaemon.logger.error(
                "Error loading configuration from preferences: \(String(unredacted: error), privacy: .public)"
            )
            throw error
        }
    }

    /// Enforces security configuration if running on Apple infrastructure (as opposed to e.g. the VRE) with a
    /// 'customer' security policy
    mutating func enforceAppleInfrastructureSecurityConfig() throws {
        ConfigurationDaemon.logger.info("Enforcing Apple infrastructure security config")
        guard let configurationService = self.configurationService else {
            ConfigurationDaemon.logger.warning("No Configuration service configuration present")
            return
        }

        if configurationService.metadataURL.scheme?.lowercased() != "https" {
            ConfigurationDaemon.logger
                .error(
                    "Error: metadataURL scheme is not 'https' (\(configurationService.metadataURL.scheme ?? "<nil>", privacy: .public))"
                )
            throw ConfigurationDConfigurationError.disallowedURLScheme
        }

        if configurationService.storageURL.scheme?.lowercased() != "https" {
            ConfigurationDaemon.logger
                .error(
                    "Error: storageURL scheme is not 'https' (\(configurationService.storageURL.scheme ?? "<nil>", privacy: .public))"
                )
            throw ConfigurationDConfigurationError.disallowedURLScheme
        }
    }

    static func fromPreferences(
        _ preferences: CFPreferences,
        secureConfigLoader: SecureConfigLoader
    ) throws -> ConfigurationDConfiguration {
        let decoder = CFPreferenceDecoder()
        var configuration = try decoder.decode(ConfigurationDConfiguration.self, from: preferences)

        let secureConfig: SecureConfig
        do {
            ConfigurationDaemon.logger.info("Loading secure config from SecureConfigDB")
            secureConfig = try secureConfigLoader.load()
        } catch {
            ConfigurationDaemon.logger.error(
                "Error loading secure config: \(String(unredacted: error), privacy: .public)"
            )
            throw error
        }

        if secureConfig.shouldEnforceAppleInfrastructureSecurityConfig {
            try configuration.enforceAppleInfrastructureSecurityConfig()
        }

        return configuration
    }
}

extension ReleaseInfo {
    init(_ releaseInfo: ConfigurationDConfiguration.ReleaseInfo, instance: String?) {
        self.init(
            project: releaseInfo.project,
            environment: releaseInfo.environment,
            release: releaseInfo.release,
            instance: instance
        )
    }
}

extension Fetcher.SchedulingConfiguration {
    init(_ schedule: ConfigurationDConfiguration.FetcherSchedule) {
        self.init(
            tickInterval: .seconds(schedule.tickInterval),
            maximumToleranceRatio: schedule.maximumToleranceRatio,
            minimumInterTickDelayRatio: schedule.minimumInterTickDelayRatio
        )
    }
}

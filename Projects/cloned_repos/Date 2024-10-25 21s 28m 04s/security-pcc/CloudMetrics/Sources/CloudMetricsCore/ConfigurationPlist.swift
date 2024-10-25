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
//  ConfigurationPlist.swift
//  CloudMetricsDaemon
//
//  Created by Andrea Guzzo on 10/23/23.
//

import Foundation
import os

private let logger = Logger(subsystem: kCloudMetricsLoggingSubsystem, category: "ConfigurationPlist")

internal struct ConfigPlistDestination: Codable {
    private enum CodingKeys: String, CodingKey {
        case publishInterval = "PublishInterval"
        case workspace = "Workspace"
        case namespace = "Namespace"
        case clients = "Clients"
    }

    internal var publishInterval: Int
    internal var workspace: String
    internal var namespace: String
    internal var clients: [String]
}

internal struct ConfigPlistDefaultDestination: Codable {
    private enum CodingKeys: String, CodingKey {
        case workspace = "Workspace"
        case namespace = "Namespace"
    }

    internal var workspace: String
    internal var namespace: String
    internal var publishInterval: Int?
}

internal struct ConfigPlistOpenTelemtryEndpoint: Codable {
    private enum CodingKeys: String, CodingKey {
        case hostname = "Hostname"
        case port = "Port"
        case disableMtls = "DisableMtls"
    }

    internal var hostname: String
    internal var port: Int
    internal var disableMtls: Bool
}

internal struct ConfigPlistCertificateConfig: Codable {
    private enum CodingKeys: String, CodingKey {
        case mtlsPrivateKeyData = "MtlsPrivateKeyData"
        case mtlsCertificateChainData = "MtlsCertificateChainData"
    }
    internal var mtlsPrivateKeyData: String?
    internal var mtlsCertificateChainData: String?
}

// swiftlint:disable discouraged_optional_boolean
internal struct ConfigurationPlist: Codable {
    private enum CodingKeys: String, CodingKey {
        case destinations = "Destinations"
        case defaultDestination = "DefaultDestination"
        case globalLabels = "GlobalLabels"
        case useOpenTelemetryBackend = "UseOpenTelemetryBackend"
        case requireAllowList = "RequireAllowList"
        case openTelemetryEndpoint = "OpenTelemetryEndpoint"
        case auditLists = "AuditLists"
        case localCertificateConfig = "LocalCertificateConfig"
        case defaultHistogramBuckets = "DefaultHistogramBuckets"
		case auditLogThrottleIntervalSeconds = "AuditLogThrottleIntervalSeconds"
    }

    internal var destinations: [ConfigPlistDestination] = []
    internal var defaultDestination: ConfigPlistDefaultDestination?
    internal var globalLabels: [String: String] = [:]
    internal var useOpenTelemetryBackend: Bool?
    internal var requireAllowList: Bool?
    internal var openTelemetryEndpoint: ConfigPlistOpenTelemtryEndpoint?
    internal var auditLists: AuditListsPlist?
    internal var localCertificateConfig: ConfigPlistCertificateConfig?
    internal var defaultHistogramBuckets: [Double] = []
	internal var auditLogThrottleIntervalSeconds: Int = kCloudMetricsAuditLogThrottleIntervalDefault

    internal init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.destinations = try container.decode([ConfigPlistDestination].self, forKey: .destinations)
        self.defaultDestination = try container.decodeIfPresent(ConfigPlistDefaultDestination.self,
                                                                forKey: .defaultDestination)
        self.globalLabels = try container.decode([String: String].self, forKey: .globalLabels)
        self.useOpenTelemetryBackend = try container.decodeIfPresent(Bool.self, forKey: .useOpenTelemetryBackend)
        self.requireAllowList = try container.decodeIfPresent(Bool.self, forKey: .requireAllowList)
        self.openTelemetryEndpoint = try container.decodeIfPresent(ConfigPlistOpenTelemtryEndpoint.self,
                                                                   forKey: .openTelemetryEndpoint)
        self.auditLists = try container.decodeIfPresent(AuditListsPlist.self, forKey: .auditLists)
        self.localCertificateConfig = try container.decodeIfPresent(ConfigPlistCertificateConfig.self, forKey: .localCertificateConfig)
        self.defaultHistogramBuckets = try container.decodeIfPresent([Double].self, forKey: .defaultHistogramBuckets) ?? []
        try self.validateDestinations()
    }

    private init() {
    }

    internal static func createFromCFPrefs() throws -> ConfigurationPlist {
        var configPlist = ConfigurationPlist()

        guard let destinationsPref = try preferencesArrayOfDict("Destinations") else {
            logger.error("Error reading destinations from Preferences")
            throw ConfigurationError.noDestinations
        }

        logger.info("\(destinationsPref, privacy: .public)")
        for dest in destinationsPref {
            guard let publishInterval = dest["PublishInterval"] as? Int,
                  let workspace = dest["Workspace"] as? String,
                  let namespace = dest["Namespace"] as? String,
                  let clients = dest["Clients"] as? [String]
            else {
                logger.error("Error decoding a destination from Preferences")
                throw ConfigurationError.invalidDestination(dest.description)
            }

            let configdest = ConfigPlistDestination(
                publishInterval: publishInterval,
                workspace: workspace,
                namespace: namespace,
                clients: clients
            )
            configPlist.destinations.append(configdest)
        }

        if let defaultDestinationPref = try preferencesDictionaryValue("DefaultDestination") {
            guard let defworkspace = defaultDestinationPref["Workspace"] as? String,
                  let defnamespace = defaultDestinationPref["Namespace"] as? String
            else {
                logger.error("Error decoding defaultDestination from Preferences")
                throw ConfigurationError.invalidDestination(defaultDestinationPref.description)
            }
            configPlist.defaultDestination = ConfigPlistDefaultDestination(workspace: defworkspace,
                                                                           namespace: defnamespace)
        } else {
            logger.error("No defaultDestination found in Preferences")
        }

        if let globalLabelsOverrides = try preferencesDictionaryValue("GlobalLabels") {
            configPlist.globalLabels = Dictionary(uniqueKeysWithValues: globalLabelsOverrides.compactMap {
                if let key = $0 as? String, let value = $1 as? String {
                    return (key, value)
                } else {
                    return nil
                }
            })
        } else {
            logger.debug("No global labels found in Preferences")
        }

        if let requireAllowList = try preferencesBoolValue("RequireAllowList") {
            configPlist.requireAllowList = requireAllowList
        }
        
        if let localCertificateConfig: [String: String] = try preferencesDictionaryValue("LocalCertificateConfig") {
            configPlist.localCertificateConfig = ConfigPlistCertificateConfig()
            configPlist.localCertificateConfig?.mtlsPrivateKeyData = localCertificateConfig["MtlsPrivateKeyData"]
            configPlist.localCertificateConfig?.mtlsCertificateChainData = localCertificateConfig["MtlsCertificateChainData"]
        }

        if let useOpenTelemetryBackend = try preferencesBoolValue("UseOpenTelemetryBackend") {
            configPlist.useOpenTelemetryBackend = useOpenTelemetryBackend

            if let openTelemtryEndpoint = try preferencesDictionaryValue("OpenTelemetryEndpoint") {
                guard let hostname = openTelemtryEndpoint["Hostname"] as? String,
                      let port = openTelemtryEndpoint["Port"] as? Int,
                      let disableMtls = openTelemtryEndpoint["DisableMtls"] as? Bool else {
                    logger.error("Error decoding OpenTelemetryEndpoint from Preferences")
                    throw ConfigurationError.invalidOpenTelemetryEndpoint(openTelemtryEndpoint.description)
                }
                logger.log("DisableMTLS: \(disableMtls, privacy: .public) for \(openTelemtryEndpoint, privacy: .public)")
                configPlist.openTelemetryEndpoint = ConfigPlistOpenTelemtryEndpoint(hostname: hostname,
                                                                                    port: port,
                                                                                    disableMtls: disableMtls)
            } else if useOpenTelemetryBackend {
                logger.log("UseOpenTelemetryBackend is true but no endpoint has been configured.")
            }
        }

        if let auditLists = try preferencesDictionaryValue("AuditLists") {
            if let allowedMetrics = auditLists["AllowedMetrics"] as? [NSDictionary] {
                let metricRules = allowedMetrics.map { ruleDict in
                    let label: String = checkAndConvert(dict: ruleDict,
                                                        key: "Label",
                                                        defaultValue: "")
                    let minUpdateInterval: Double = checkAndConvert(dict: ruleDict,
                                                                    key: "MinUpdateInterval",
                                                                    defaultValue: 0)
                    let minPublishInterval: Double = checkAndConvert(dict: ruleDict,
                                                                     key: "MinPublishInterval",
                                                                     defaultValue: 0)
                    let type: String? = checkAndConvert(dict: ruleDict,
                                                        key: "Type",
                                                        defaultValue: nil)
                    let destinations: [String] = checkAndConvert(dict: ruleDict,
                                                                 key: "Destinations",
                                                                 defaultValue: [])
                    let client: String = checkAndConvert(dict: ruleDict,
                                                         key: "Client",
                                                         defaultValue: "")
                    let dimensions: [String: [String]] = checkAndConvert(dict: ruleDict,
                                                                         key: "Dimensions",
                                                                         defaultValue: [:])

                    let rulePlist = MetricRulePlist(
                        client: client,
                        label: label,
                        minUpdateInterval: minUpdateInterval,
                        minPublishInterval: minPublishInterval,
                        type: type,
                        destinations: destinations,
                        dimensions: dimensions)
                    return rulePlist
                }
                var ignoredMetrics: [String: [String]] = [:]
                if let configuredIgnoredMetrics = auditLists["IgnoredMetrics"] as? [String: [String]] {
                    ignoredMetrics = configuredIgnoredMetrics
                }
                configPlist.auditLists = AuditListsPlist(allowedMetrics: metricRules,
                                                         ignoredMetrics: ignoredMetrics)
            } else {
                throw ConfigurationError.auditListDecodingError
            }
        }

        if let buckets: [Double] = try preferencesArrayValue("DefaultHistogramBuckets") {
            configPlist.defaultHistogramBuckets = buckets
        }

		if let throttleInterval = try preferencesIntegerValue("AuditLogThorttleIntervalSeconds") {
			configPlist.auditLogThrottleIntervalSeconds = throttleInterval
		}
        try configPlist.validateDestinations()
        return configPlist
    }

    internal mutating func validateDestinations() throws {
        guard let defaultDestination = self.defaultDestination else {
            throw ConfigurationError.defaultDestinationNotConfigured
        }
        var defaultDestinationFound = false
        var dedupDestinations: Set<String> = []
        for destination in destinations {
            if destination.namespace == defaultDestination.namespace,
               destination.workspace == defaultDestination.workspace {
                var newDefaultDestination = defaultDestination
                newDefaultDestination.publishInterval = destination.publishInterval
                self.defaultDestination = newDefaultDestination
                defaultDestinationFound = true
            }
            let uniqueId = "\(destination.workspace)/\(destination.namespace)"
            if dedupDestinations.contains(uniqueId) {
                throw ConfigurationError.duplicateDestinationConfiguration(uniqueId)
            }
            dedupDestinations.insert(uniqueId)
        }
        if defaultDestinationFound == false {
            throw ConfigurationError.defaultDestinationNotFound
        }
    }
}

private func checkAndConvert<T>(dict: NSDictionary, key: String, defaultValue: T) -> T {
    if dict[key] != nil {
        if let value = dict[key] as? T {
            return value
        } else {
            logger.error("Can't convert value \(String(describing: dict[key]), privacy: .public) as \(T.self, privacy: .public)")
        }
        return defaultValue
    }
    return defaultValue
}

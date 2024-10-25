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

/*swift tabwidth="4" prefertabs="false"*/
//
//  CloudMetricsConfiguration.swift
//  CloudMetricsDaemon
//
//  Created by Dhanasekar Thangavel on 1/27/23.
//

// swiftlint:disable file_length

#if canImport(cloudOSInfo)
@_weakLinked import cloudOSInfo
#endif
import MobileGestaltPrivate
import NIOSSL
import os

internal let kDefaultPublishingInternal = 60

internal enum ConfigurationError: Error, CustomStringConvertible {
    case defaultDestinationNotConfigured
    case defaultDestinationNotFound
    case duplicateDestinationConfiguration(String)
    case invalidTLSConfiguration
    case decodingError
    case noDestinations
    case invalidDestination(String)
    case invalidOpenTelemetryEndpoint(String)
    case unkownCloudMetricsType(String)
    case auditListDecodingError
    case invalidCertificateConfiguration(String)
    case MosaicNotSupported

    internal var description: String {
        switch self {
        case .defaultDestinationNotConfigured:
            return "Default Destination Not Configured"
        case .defaultDestinationNotFound:
            return "Default Destination not present in the Destinations array"
        case .duplicateDestinationConfiguration(let destination):
            return "Duplicate Destination \(destination)"
        case .invalidTLSConfiguration:
            return "Invalid TLS configuration"
        case .decodingError:
            return "Error decoding"
        case .noDestinations:
            return "No destinations configured"
        case .invalidDestination(let destination):
            return "Invalid destination: \(destination)"
        case .invalidOpenTelemetryEndpoint(let endpoint):
            return "Invalid OpenTelemetry endpoint: \(endpoint)"
        case .unkownCloudMetricsType(let typeString):
            return "Unknown type: \(typeString)"
        case .auditListDecodingError:
            return "Can't decode the provided AuditList"
        case .invalidCertificateConfiguration(let error):
            return error
        case .MosaicNotSupported:
            return "Mosaic not supported at build time"
        }
    }
}

internal class CloudMetricsCertConfig: @unchecked Sendable {
    internal let mtlsPrivateKey: NIOSSLPrivateKey?
    internal let mtlsCertificateChain: [NIOSSLCertificate]
    internal let mtlsTrustRoots: NIOSSLTrustRoots
    internal let hostName: String?

    internal init(mtlsPrivateKey: NIOSSLPrivateKey? = nil,
                  mtlsCertificateChain: [NIOSSLCertificate] = [],
                  mtlsTrustRoots: NIOSSLTrustRoots = .default,
                  hostName: String? = nil) {
        self.mtlsPrivateKey = mtlsPrivateKey
        self.mtlsCertificateChain = mtlsCertificateChain
        self.mtlsTrustRoots = mtlsTrustRoots
        self.hostName = hostName
    }
}

internal final class CloudMetricsDestination: Equatable, CustomStringConvertible, @unchecked Sendable {
    internal let namespace: String
    internal let workspace: String
    internal let publishInterval: Int // seconds
    // swiftlint:disable:next identifier_name
    internal var _certificates: CloudMetricsCertConfig
    private let lock = OSAllocatedUnfairLock()

    internal var id: String {
        "\(workspace)/\(namespace)/\(publishInterval)"
    }

    internal var certificates: CloudMetricsCertConfig {
        lock.lock()
        let certificates = self._certificates
        lock.unlock()
        return certificates
    }

    internal var description: String {
        "Destination{\(id)}"
    }

    internal init(workspace: String,
                  namespace: String,
                  certificates: CloudMetricsCertConfig,
                  publishInterval: Int = kDefaultPublishingInternal) {
        self.workspace = workspace
        self.namespace = namespace
        self._certificates = certificates
        self.publishInterval = publishInterval
    }

    internal init(id: String) throws {
        let components = id.split(separator: "/")
        if components.count < 2 {
            throw ConfigurationError.invalidDestination(id)
        }
        self.workspace = String(components[0])
        self.namespace = String(components[1])
        if components.count > 2, let interval = Int(components[3]) {
            self.publishInterval = interval
        } else {
            self.publishInterval = kDefaultPublishingInternal
        }
        self._certificates = CloudMetricsCertConfig()
    }

    internal static func == (lhs: CloudMetricsDestination, rhs: CloudMetricsDestination) -> Bool {
        (lhs.namespace == rhs.namespace &&
         lhs.workspace == rhs.workspace)
    }

    internal func updateCertificates(_ certificates: CloudMetricsCertConfig) {
        lock.lock()
        self._certificates = certificates
        lock.unlock()
    }
}

internal struct CloudMetricsOTEndpoint: Equatable {
    internal let hostname: String
    internal let port: Int
    internal let disableMtls: Bool
    internal init(hostname: String, port: Int, disableMtls: Bool = false) {
        self.hostname = hostname
        self.port = port
        self.disableMtls = disableMtls
    }
}

internal struct CloudMetricsFilterRule: Sendable {
    internal let client: String
    internal let label: String
    internal let minUpdateInterval: TimeInterval
    internal let minPublishInterval: TimeInterval
    internal let type: CloudMetricType?
    internal let destinations: [CloudMetricsDestination]
    internal let dimensions: [String: [String]]

    internal init(client: String,
                  label: String,
                  minUpdateInterval: TimeInterval,
                  minPublishInterval: TimeInterval,
                  type: CloudMetricType?,
                  destinations: [CloudMetricsDestination],
                  dimensions: [String: [String]]) {
        self.client = client
        self.label = label
        self.minUpdateInterval = minUpdateInterval
        self.minPublishInterval = minPublishInterval
        self.type = type
        self.destinations = destinations
        self.dimensions = dimensions
    }

    internal init(fromPlist: MetricRulePlist) throws {
        self.client = fromPlist.client
        self.label = fromPlist.label
        self.minUpdateInterval = fromPlist.minUpdateInterval
        self.minPublishInterval = fromPlist.minPublishInterval
        if let typeString = fromPlist.type {
            if let type = CloudMetricType(rawValue: typeString) {
                self.type = type
            } else {
                throw ConfigurationError.unkownCloudMetricsType(typeString)
            }
        } else {
            self.type = nil
        }

        self.destinations = try fromPlist.destinations.compactMap { destinationId in
            let destination = try CloudMetricsDestination(id: destinationId)
            return destination
        }
        self.dimensions = fromPlist.dimensions
    }
}

private struct SystemCryptexVersionPlist: Codable {
    private enum CodingKeys: String, CodingKey {
        case productBuildVersion = "ProductBuildVersion"
    }

    var productBuildVersion: String
}

internal struct CloudMetricsConfiguration {
    private let configPlist: ConfigurationPlist
    private var tlsCerts: CloudMetricsCertConfig?
    private let logger = Logger(subsystem: kCloudMetricsLoggingSubsystem, category: "Configuration")
    private var auditLists: AuditListsPlist?

    internal var useOpenTelemetry: Bool {
        configPlist.useOpenTelemetryBackend ?? true
    }
    
    internal var localCertificateConfig: CloudMetricsCertConfig? {
        if let certConfig = configPlist.localCertificateConfig {
            var mtlsPrivateKey: NIOSSLPrivateKey? = nil
            var mtlsCertificateChain: NIOSSLCertificate? = nil
            var mtlsTrustRoots: NIOSSLTrustRoots? = nil
            if let privateKeyData = certConfig.mtlsPrivateKeyData,
               let certificateChainData = certConfig.mtlsCertificateChainData {

                do {
                    logger.info("Loading private key from the local certificate configuration")

                    let privateKeyBytes: [UInt8] = Array(privateKeyData.utf8)
                    mtlsPrivateKey = try .init(bytes: privateKeyBytes, format: .pem)

                    logger.info("Loading cert chain  from the local certificate configuration")
                    let certificateChainBytes: [UInt8] = Array(certificateChainData.utf8)
                    mtlsCertificateChain = try .init(bytes: certificateChainBytes, format: .pem)

                    let cryptexMountPoint = ProcessInfo.processInfo.environment["CRYPTEX_MOUNT_PATH"] ?? ""
                    var trustRootsRelativePath = "/usr/share/cloudmetricsd/mosaic_trustroot.pem"
#if os(iOS)
                    if MobileGestalt.current.isComputeController {
                        trustRootsRelativePath = "/usr/share/cloudmetricsd_bmc/mosaic_trustroot.pem"
                    }
#endif
                    let trustRootsFilePath = "\(cryptexMountPoint)\(trustRootsRelativePath)"
                    mtlsTrustRoots = NIOSSLTrustRoots.certificates([
                        try NIOSSLCertificate(file: trustRootsFilePath, format: .pem),
                    ])
                } catch {
                    logger.error("Privatekey file or certchain data invalid, disabling the certificate configuration")
                    return nil
                }
            }

            if mtlsPrivateKey == nil {
                logger.debug("No PrivateKey configured")
            }
            if mtlsCertificateChain == nil {
                logger.debug("No CertificateChain configured")
            }

            return CloudMetricsCertConfig(mtlsPrivateKey: mtlsPrivateKey,
                                          mtlsCertificateChain: (mtlsCertificateChain != nil) ? [mtlsCertificateChain!] : [],
                                          mtlsTrustRoots: mtlsTrustRoots ?? .default)
        }
        return nil
    }

    internal var requireAllowList: Bool {
        // TODO change default behavior:
        // rdar://121395595 (Ability to handle legacy use cases with allow-lists for both metrics and log.)
        configPlist.requireAllowList ?? false
    }

    internal var defaultDestination: CloudMetricsDestination? {
        guard let destination = configPlist.defaultDestination,
              let tlsCerts = self.tlsCerts,
              let publishInterval = destination.publishInterval
        else {
            return nil
        }
        return CloudMetricsDestination(workspace: destination.workspace,
                                       namespace: destination.namespace,
                                       certificates: tlsCerts,
                                       publishInterval: publishInterval)
    }

    internal var openTelemetryEndpoint: CloudMetricsOTEndpoint? {
        if let endpoint = configPlist.openTelemetryEndpoint {
            return CloudMetricsOTEndpoint(hostname: endpoint.hostname,
                                          port: endpoint.port,
                                          disableMtls: endpoint.disableMtls)
        }
        return nil
    }

    internal var metricsAllowList: [CloudMetricsFilterRule] {
        self.auditLists?.allowedMetrics.map { value in
            var metricType: CloudMetricType?
            if let type = value.type {
                if let cmType = CloudMetricType(rawValue: type) {
                    metricType = cmType
                } else {
                    logger.error("Unknown metric type: \(type)")
                }
            }
            return CloudMetricsFilterRule(client: value.client,
                                          label: value.label,
                                          minUpdateInterval: value.minUpdateInterval,
                                          minPublishInterval: value.minPublishInterval,
                                          type: metricType,
                                          destinations: value.destinations.compactMap { destinationId in
                                                do {
                                                    let destination = try CloudMetricsDestination(id: destinationId)
                                                    return destination
                                                } catch {
                                                    self.logger.error("Invalid destination ID: \(destinationId, privacy: .public)")
                                                }
                                                return nil
                                          },
                                           dimensions: value.dimensions)
        } ?? []
    }

    internal var metricsIgnoreList: [String: [String]] {
        self.auditLists?.ignoredMetrics ?? [:]
    }

    internal var globalLabels: [String: String] {
        var globalLabels: [String: String] = self.configuredLabels()
        if #_hasSymbol(CloudOSInfoProvider.self) {
            let cloudOSInfo = CloudOSInfoProvider()
            if #_hasSymbol(cloudOSInfo.observabilityLabels) {
                do {
                    let observabilityLabels = try cloudOSInfo.observabilityLabels()
                    globalLabels.merge(observabilityLabels) { (_, new) in new }
                } catch {
                    logger.info("Can't load observability labels: \(error, privacy: .public)")
                }
            }
        }
        let nodeUDID = MobileGestalt.current.uniqueDeviceID
        if nodeUDID == nil {
            logger.error("Can't get a valid UDID")
        }
        globalLabels["_udid"] = nodeUDID ?? ""

        let hwModel = MobileGestalt.current.hwModelStr
        if hwModel == nil {
            logger.error("Can't get a valid HWModel string")
        }
        globalLabels["_hwmodel"] = hwModel ?? ""

        let systemCryptexVersion = self.systemCryptexVersion
        logger.debug("System Cryptex version: \(systemCryptexVersion ?? "unknown", privacy: .public)")
        globalLabels["_systemcryptexversion"] = systemCryptexVersion ?? ""

        globalLabels["_projectid"] = ""
        do {
            // Lookup the default projectID configured by darwin-init in CFPrefs.
            let cloudUsageTrackingDomain = "com.apple.acsi.cloudusagetrackingd"
            if let projectId = try preferencesStringValue("defaultProjectID", domain: cloudUsageTrackingDomain) {
                globalLabels["_projectid"] = projectId
            }
        } catch {
            logger.error("Can't get the project ID: \(error, privacy: .public)")
        }

    #if os(macOS)
        globalLabels["_type"] = "node"
    #elseif os(iOS)
        if MobileGestalt.current.isComputeController {
            globalLabels["_type"] = "bmc"
        } else {
            globalLabels["_type"] = "node"
        }
    #endif
        
        let hostName = self.tlsCerts?.hostName
        if hostName == nil {
            logger.error("Can't get the hostname.")
        }

        globalLabels["_hostname"] = hostName ?? ""
        return globalLabels
    }

    internal var systemCryptexVersion: String? {
        if #_hasSymbol(CloudOSInfoProvider.self) {
            let cloudOSInfo = CloudOSInfoProvider()
            do {
                let buildVersion = try cloudOSInfo.cloudOSBuildVersion()
                return buildVersion
            } catch {
                logger.error("unable to determine build version from deployment manifest: \(error, privacy: .public), will attempt to fallback to cryptex version.plist")
            }

            do {
                let buildVersion = try cloudOSInfo.extractVersionFromSupportCryptex()
                return buildVersion
            } catch {
                logger.error("failed to determine build version from cryptex: \(error, privacy: .public)")
            }
        }
        return nil
    }

    internal var defaultHistogramBuckets: [Double] {
        return configPlist.defaultHistogramBuckets
    }

	internal var auditLogThrottleIntervalSeconds: Int {
		return configPlist.auditLogThrottleIntervalSeconds
	}

    internal init(configurationFile: String?, auditLists: AuditListsPlist? = nil) async throws {
        // If a configfile is specified, it overrides the CFPrefs configuration
        if let configFile = configurationFile, FileManager.default.fileExists(atPath: configFile) {
            logger.info("Reading from provided configuration file: \(configFile, privacy: .public)")
            let decoder = PropertyListDecoder()
            configPlist = try decoder.decode(
                ConfigurationPlist.self,
                from: try Data(contentsOf: URL(filePath: configFile))
            )
        } else { // Second, if the plist does not exists look at the CFPref settings
            logger.info("Reading from CFPrefs")
            let configPlist = try ConfigurationPlist.createFromCFPrefs()
            self.configPlist = configPlist
            let labels = self.configuredLabels()
            logger.info("""
                destinations='\(configPlist.destinations, privacy: .private)', \
                defaultDestination='\(String(describing: configPlist.defaultDestination))', \
                globalLabels='\(String(describing: labels))', \
                useOpenTelemetry='\(String(describing: configPlist.useOpenTelemetryBackend))'
                """)
        }

        if let auditListsFromInit = auditLists {
            self.auditLists = auditListsFromInit
        }
#if DEBUG
        if self.auditLists == nil, let auditListsFromConfig = configPlist.auditLists {
            self.auditLists = auditListsFromConfig
        }
#endif

        logger.info("Loading certificates for Mosaic")
        
        // check if the useCertMgrCertKeyFile so we load the cert from key and cert file instead of from the narrative.
        // This is primarily used for development testing puprose.
        if let certificateConfig = self.localCertificateConfig {
            logger.info("Loading certificates from cfprefs \(String(describing: certificateConfig))")
            self.tlsCerts = certificateConfig
        }
        else {
            logger.info("Loading narrative certificates from keychain")
            self.tlsCerts = try await loadTLSCerts()
        }
    }

    internal func configuredLabels() -> [String: String] {
        configPlist.globalLabels
    }

    internal func destinationConfigurations() throws -> [CloudMetricsDestination] {
        guard let tlsCerts = self.tlsCerts,
              let privateKey = tlsCerts.mtlsPrivateKey else {
            throw ConfigurationError.invalidTLSConfiguration
        }
        return configPlist.destinations.map { destination in
            CloudMetricsDestination(
                workspace: destination.workspace,
                namespace: destination.namespace,
                certificates: CloudMetricsCertConfig(mtlsPrivateKey: privateKey,
                                                     mtlsCertificateChain: tlsCerts.mtlsCertificateChain,
                                                     mtlsTrustRoots: tlsCerts.mtlsTrustRoots,
                                                     hostName: tlsCerts.hostName),
                publishInterval: .init(destination.publishInterval)
            )
        }
    }

    internal func clientDestinations() throws -> [String: CloudMetricsDestination] {
        guard let certsConfig = self.tlsCerts else {
            throw ConfigurationError.invalidTLSConfiguration
        }

        return Dictionary(configPlist.destinations.flatMap { destination in
            destination.clients.map { client in
                (client, CloudMetricsDestination(workspace: destination.workspace,
                                                 namespace: destination.namespace,
                                                 certificates: certsConfig,
                                                 publishInterval: destination.publishInterval))
            }
        }) { _, last in last }
    }
}

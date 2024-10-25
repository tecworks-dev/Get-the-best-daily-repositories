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
//  OpenTelemetryPublisher.swift
//  CloudMetricsDaemon
//
//  Created by Andrea Guzzo on 9/28/23.
//

import Foundation
import GRPC
import Logging
import NIO
import NIOHPACK
import NIOSSL
import OpenTelemetryApi
import OpenTelemetryProtocolExporterCommon
import OpenTelemetryProtocolExporterGrpc
import OpenTelemetrySdk
import os

private let kDefaultOTLPHost = "localhost"
private let kDefaultOTLPPort = 4_317

internal class OpenTelemetryPublisher: CloudMetricsPublisher {
    private let cloudMetricsConfiguration: CloudMetricsConfiguration
    private let defaultDestination: CloudMetricsDestination
    private let clientDestinations: [String: CloudMetricsDestination]
    private var metricsStores: [String: (OpenTelemetryStore, StableMeterProviderSdk)] = [:]
    private let collectorEndpoint: CloudMetricsOTEndpoint
    private let metricsFilter: MetricsFilter
    private let logger = Logger(subsystem: kCloudMetricsLoggingSubsystem, category: "OpenTelemetryPublisher")

    internal init(configuration: CloudMetricsConfiguration, metricsFilter: MetricsFilter) throws {
        self.cloudMetricsConfiguration = configuration

        guard let defaultDestination = configuration.defaultDestination else {
            throw ConfigurationError.defaultDestinationNotConfigured
        }
        self.defaultDestination = defaultDestination

        clientDestinations = try configuration.clientDestinations()

        self.metricsFilter = metricsFilter

        collectorEndpoint = configuration.openTelemetryEndpoint ??
            CloudMetricsOTEndpoint(hostname: kDefaultOTLPHost, port: kDefaultOTLPPort, disableMtls: true)

        for (_, destination) in clientDestinations {
            try setupConnections(destination: destination)
        }

        // make sure the default destination is also initialized
        try setupConnections(destination: defaultDestination)
    }

    private func setupConnections(destination: CloudMetricsDestination) throws {
        if metricsStores[destination.id] != nil {
            // nothing to do, this destination has been already configured
            return
        }

        var clientConfiguration = ClientConnection.Configuration.default(
            target: .hostAndPort(collectorEndpoint.hostname, collectorEndpoint.port),
            eventLoopGroup: MultiThreadedEventLoopGroup(numberOfThreads: 1)
        )
        if let privateKey = destination.certificates.mtlsPrivateKey, collectorEndpoint.disableMtls == false {
            let certificateChain = destination.certificates.mtlsCertificateChain.map {
                NIOSSLCertificateSource.certificate($0)
            }
            let tlsConfig = GRPCTLSConfiguration.makeClientConfigurationBackedByNIOSSL(
                certificateChain: certificateChain,
                privateKey: .privateKey(privateKey),
                trustRoots: destination.certificates.mtlsTrustRoots,
                certificateVerification: .noHostnameVerification)
            clientConfiguration.tlsConfiguration = tlsConfig
        } else {
            logger.info("No mTLS configuration for destination \(destination.id, privacy: .private)")
        }
        let client = ClientConnection(configuration: clientConfiguration)

        logger.debug("Setup OTLP client connection: \(String(describing: client), privacy: .public)")
        // Initialize OpenTelemtry
        let otlpHeaders: [(String, String)] =
        [
            ("X-MOSAIC-WORKSPACE", destination.workspace),
            ("X-MOSAIC-NAMESPACE", destination.namespace),
        ]
        let otlpConfiguration = OtlpConfiguration(headers: otlpHeaders)
        let otlpMetricExporter = OpenTelemetryMetricExporter(
            channel: client,
            config: otlpConfiguration,
            aggregationTemporalitySelector: AggregationTemporality.deltaPreferred(),
            metricsFilter: metricsFilter,
            destination: destination)

        let metricsReader = OpenTelemetryPeriodicMetricReader(exporter: otlpMetricExporter,
                                                              exportInterval: TimeInterval(destination.publishInterval))
        let cloudMetricsAggregation = CloudMetricsAggregation(histogramBuckets: cloudMetricsConfiguration.defaultHistogramBuckets)
        let metricsView = StableView.builder().withAggregation(aggregation: cloudMetricsAggregation).build()

        let meterProvider = StableMeterProviderSdk.builder()
            .registerMetricReader(reader: metricsReader)
            .registerView(selector: InstrumentSelector.builder().setInstrument(name: ".*").build(), view: metricsView)
            .build()
        let store = OpenTelemetryStore(
            meter: meterProvider.meterBuilder(name: "CloudMetrics").build(),
            globalLabels: self.cloudMetricsConfiguration.globalLabels)
        metricsReader.store = store
        metricsStores[destination.id] = (store, meterProvider)

        if cloudMetricsConfiguration.localCertificateConfig == nil, collectorEndpoint.disableMtls == false {
            try registerCertificateRenewalHandler(destination: destination)
        }
    }

    internal func getMetricsStore(for client: String) throws -> MetricsStore? {
        // use the default destination if none is configured for this client.
        let destination = getDestination(for: client)

        guard let (store, _) = metricsStores[destination.id] else {
            // There must always be a default store and a client must always point to one.
            logger.error("Could not find metrics store. client='\(client, privacy: .public)' destination='\(destination.id, privacy: .private)'")
            return nil
        }

        return store
    }

    internal func getDestination(for client: String) -> CloudMetricsDestination {
        clientDestinations[client] ?? defaultDestination
    }

    internal func run() async throws {
        // Opentelemtry handles publishing internally (using its PushMetricController and
        // running the OtlpMetricExporter we pass to the meterProvider).
        return await withUnsafeContinuation { _ in }
    }

    internal func shutdown() async throws {
        metricsStores.removeAll()
    }

    private func registerCertificateRenewalHandler(destination: CloudMetricsDestination) throws {
        logger.debug("Setting up RenewCertificate callback")
        let currentCert = destination.certificates.mtlsCertificateChain[0]
        let certExpiryHandler = try NICCertExpiryHandler() { newCert in
            self.logger.debug("""
                RenewCertificate callback called. \
                Reconfiguring destination: \(String(describing: destination), privacy: .private).
                """)

            guard let (_, meterProvider) = self.metricsStores[destination.id] else {
                throw OpenTelemetryStoreError.noMeterForDestination(destinationID: destination.id)
            }

            if meterProvider.forceFlush() == .failure {
                self.logger.error("Can't flush metrics before cert renewal")
            }

            if meterProvider.shutdown() == .failure {
                self.logger.error("Can't shutdown the meterProvider before cert renewal")
            }

            guard let privateKey = newCert.mtlsPrivateKey else {
                throw NarrativeIdentityError.privateKeyMissing("newCert doesn't contain a private key")
            }

            let newCertConfig = CloudMetricsCertConfig(mtlsPrivateKey: privateKey,
                                                       mtlsCertificateChain: newCert.mtlsCertificateChain,
                                                       mtlsTrustRoots: newCert.mtlsTrustRoots,
                                                       hostName: newCert.hostName)

            self.metricsStores[destination.id] = nil
            destination.updateCertificates(newCertConfig)
            try self.setupConnections(destination: destination)

            self.logger.debug("""
                Successfully renewed the certificate for destination: \(String(describing: destination), privacy: .private).
                """)
        }
        certExpiryHandler.registerCertRenewalNotification()
    }
}

// Sendability is ensured by synchronising internally.
extension OpenTelemetryPublisher: @unchecked Sendable {}

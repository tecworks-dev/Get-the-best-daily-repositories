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

import CloudBoardAttestationDAPI
import CloudBoardCommon
import CloudBoardIdentity
import CloudBoardJobAuthDAPI
import CloudBoardLogging
import CloudBoardMetrics
import CloudBoardPlatformUtilities
import Foundation
import GRPCClientConfiguration
import InternalGRPC
import Logging
import NIOCore
import NIOHTTP2
import NIOTLS
import NIOTransportServices
import os
import Security
import Tracing

/// Central coordinator daemon interacting with PCC Gateway to provide node attestations and load status as well as
/// receive and respond with encrypted requests and responses. It also provides an endpoint for the workload controller
/// to signal readiness of the workload and to provide service discovery registration metadata that cloudboardd
/// announces to Service Discovery.
public actor CloudBoardDaemon {
    enum InitializationError: Error {
        case missingAuthTokenSigningKeys
    }

    public static let logger: os.Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "cloudboardd"
    )

    @TaskLocal
    public static var rpcID: UUID = .zero

    @TaskLocal
    public static var requestTrackingID: String = ""

    static let metricsClientName = "cloudboardd"
    private static let watchdogProcessName = "cloudboardd"
    private let metricsSystem: MetricsSystem
    private let tracer: any Tracer

    private let healthMonitor: ServiceHealthMonitor
    private let heartbeatPublisher: HeartbeatPublisher?
    private let requestFielderManager: RequestFielderManager?
    private let jobHelperClientProvider: CloudBoardJobHelperClientProvider
    private let jobHelperResponseDelegateProvider: CloudBoardJobHelperResponseDelegateProvider
    private let serviceDiscovery: ServiceDiscoveryPublisher?
    private let healthServer: HealthServer?
    private let attestationProvider: AttestationProvider?
    private let config: CloudBoardDConfiguration
    private let hotProperties: HotPropertiesController?
    private let nodeInfo: NodeInfo?
    private let group: NIOTSEventLoopGroup
    private let workloadController: WorkloadController?
    private let insecureListener: Bool
    private let lifecycleManager: LifecycleManager
    private let watchdogService: CloudBoardWatchdogService?
    private let statusMonitor: StatusMonitor
    private let jobAuthClient: CloudBoardJobAuthAPIClientProtocol?
    private var exitContinuation: CheckedContinuation<Void, Never>?

    public init(configPath: String?, metricsSystem: MetricsSystem? = nil) throws {
        let nodeInfo = NodeInfo.load()
        if let isLeader = nodeInfo.isLeader {
            if !isLeader {
                CloudBoardDaemon.logger.log("Not a leader node, exiting..")
                Foundation.exit(0)
            }
        } else {
            CloudBoardDaemon.logger.error("Unable to check if node is a leader")
        }

        let config: CloudBoardDConfiguration
        if let configPath {
            Self.logger.info("Loading configuration from \(configPath, privacy: .public)")
            config = try CloudBoardDConfiguration.fromFile(path: configPath, secureConfigLoader: .real)
        } else {
            Self.logger.info("Loading configuration from preferences")
            config = try CloudBoardDConfiguration.fromPreferences()
        }
        let configJSON = try String(decoding: JSONEncoder().encode(config), as: UTF8.self)
        Self.logger.log("Loaded configuration: \(configJSON, privacy: .public)")
        try self.init(config: config, nodeInfo: nodeInfo, metricsSystem: metricsSystem)
    }

    internal init(config: CloudBoardDConfiguration, metricsSystem: MetricsSystem? = nil) throws {
        try self.init(config: config, nodeInfo: NodeInfo.load(), metricsSystem: metricsSystem)
    }

    internal init(config: CloudBoardDConfiguration, nodeInfo: NodeInfo?, metricsSystem: MetricsSystem? = nil) throws {
        self.group = NIOTSEventLoopGroup(loopCount: 1)
        let metricsSystem = metricsSystem ?? CloudMetricsSystem(clientName: CloudBoardDaemon.metricsClientName)
        self.metricsSystem = metricsSystem
        self.statusMonitor = .init(metrics: metricsSystem)
        self.tracer = RequestSummaryTracer(metrics: metricsSystem)
        self.healthMonitor = ServiceHealthMonitor()
        self.healthServer = HealthServer()
        self.requestFielderManager = try RequestFielderManager(
            prewarmedPoolSize: config.prewarming?.prewarmedPoolSize ?? 3,
            maxProcessCount: config.prewarming?.maxProcessCount ?? 3,
            metrics: self.metricsSystem,
            tracer: self.tracer
        )
        self.jobHelperClientProvider = try CloudBoardJobHelperXPCClientProvider(
            instanceManager: self.requestFielderManager!
        )
        self.jobHelperResponseDelegateProvider = CloudBoardProvider.JobHelperResponseDelegateProvider()
        self.config = config
        let hotProperties = HotPropertiesController()
        self.hotProperties = hotProperties
        self.nodeInfo = nodeInfo
        self.insecureListener = false
        self.serviceDiscovery = nil
        self.jobAuthClient = nil

        self.attestationProvider = nil

        self.workloadController = WorkloadController(
            healthPublisher: self.healthMonitor,
            metrics: self.metricsSystem
        )
        if let heartbeat = config.heartbeat {
            CloudBoardDaemon.logger.info("Heartbeat configured")
            self.heartbeatPublisher = try .init(
                configuration: heartbeat,
                identifier: ProcessInfo.processInfo.hostName,
                nodeInfo: nodeInfo,
                statusMonitor: self.statusMonitor,
                hotProperties: hotProperties,
                metrics: metricsSystem
            )
        } else {
            CloudBoardDaemon.logger.info("Heartbeat not configured")
            self.heartbeatPublisher = nil
        }

        let lifecycleManagerConfig = config.lifecycleManager ?? CloudBoardDConfiguration.LifecycleManager()
        self.lifecycleManager = LifecycleManager(
            config: .init(timeout: lifecycleManagerConfig.drainTimeout)
        )
        self.watchdogService = CloudBoardWatchdogService(processName: Self.watchdogProcessName, logger: Self.logger)
    }

    // Test entry point to allow dependency injection.
    //
    // This variant explicitly _disables_ having a service discovery integration, as
    // typically that won't be working properly in unit test scenarios unless it has
    // been deliberately set up.
    internal init(
        group: NIOTSEventLoopGroup = NIOTSEventLoopGroup(),
        healthMonitor: ServiceHealthMonitor = ServiceHealthMonitor(),
        heartbeatPublisher: HeartbeatPublisher? = nil,
        serviceDiscoveryPublisher: ServiceDiscoveryPublisher? = nil,
        jobHelperClientProvider: CloudBoardJobHelperClientProvider? = nil,
        jobHelperResponseDelegateProvider: CloudBoardJobHelperResponseDelegateProvider? = nil,
        cloudboardHealthServerInstance: HealthServer? = nil,
        lifecycleManager: LifecycleManager? = nil,
        attestationProvider: AttestationProvider? = nil,
        config: CloudBoardDConfiguration,
        hotProperties: HotPropertiesController? = nil,
        nodeInfo: NodeInfo? = nil,
        workloadController: WorkloadController?,
        insecureListener: Bool,
        metricsSystem: MetricsSystem? = nil,
        statusMonitor: StatusMonitor? = nil,
        tracer: (any Tracer)? = nil,
        jobAuthClient: CloudBoardJobAuthAPIClientProtocol? = nil
    ) throws {
        let metricsSystem = metricsSystem ?? CloudMetricsSystem(clientName: CloudBoardDaemon.metricsClientName)
        self.metricsSystem = metricsSystem
        let statusMonitor = statusMonitor ?? StatusMonitor(metrics: metricsSystem)
        self.statusMonitor = statusMonitor
        self.tracer = tracer ?? RequestSummaryTracer(metrics: metricsSystem)
        self.group = group
        self.healthMonitor = healthMonitor
        self.heartbeatPublisher = heartbeatPublisher
        self.serviceDiscovery = serviceDiscoveryPublisher
        if let jobHelperClientProvider {
            self.jobHelperClientProvider = jobHelperClientProvider
            self.requestFielderManager = nil
        } else {
            self.requestFielderManager = try RequestFielderManager(
                prewarmedPoolSize: config.prewarming?.prewarmedPoolSize ?? 0,
                maxProcessCount: config.prewarming?.maxProcessCount ?? 0,
                metrics: self.metricsSystem,
                tracer: self.tracer
            )
            self.jobHelperClientProvider = try CloudBoardJobHelperXPCClientProvider(
                instanceManager: self.requestFielderManager!
            )
        }
        if let jobHelperResponseDelegateProvider {
            self.jobHelperResponseDelegateProvider = jobHelperResponseDelegateProvider
        } else {
            self.jobHelperResponseDelegateProvider = CloudBoardProvider.JobHelperResponseDelegateProvider()
        }
        if let lifecycleManager {
            self.lifecycleManager = lifecycleManager
        } else {
            let lifecycleManagerConfig = config.lifecycleManager ?? CloudBoardDConfiguration.LifecycleManager()
            self.lifecycleManager = LifecycleManager(
                config: .init(timeout: lifecycleManagerConfig.drainTimeout)
            )
        }
        self.healthServer = cloudboardHealthServerInstance
        self.attestationProvider = attestationProvider
        self.config = config
        self.hotProperties = hotProperties
        self.nodeInfo = nodeInfo
        self.insecureListener = insecureListener
        self.workloadController = workloadController
        self.watchdogService = nil
        self.jobAuthClient = jobAuthClient
    }

    public func start() async throws {
        try await self.start(portPromise: nil, allowExit: false)
    }

    // expose more control for testing purposes
    // - a way of surfacing the GRPC server port
    // - a way to allow CloudBoardDaemon to exit
    func start(portPromise: Promise<Int, Error>?, allowExit: Bool) async throws {
        CloudBoardDaemon.logger.log("hello from cloudboardd")
        self._allowExit = allowExit

        let hotProperties = self.hotProperties
        let nodeInfo = self.nodeInfo
        let drainTimeMeasurement: OSAllocatedUnfairLock<ContinuousTimeMeasurement?> =
            OSAllocatedUnfairLock(initialState: nil)
        let cloudBoardProvider: OSAllocatedUnfairLock<CloudBoardProvider?> =
            OSAllocatedUnfairLock(initialState: nil)

        self.statusMonitor.initializing()
        let jobQuiescenceMonitor: JobQuiescenceMonitor
        await LaunchdJobHelper.cleanupManagedLaunchdJobs(logger: CloudBoardDaemon.logger)
        do {
            jobQuiescenceMonitor = JobQuiescenceMonitor(lifecycleManager: self.lifecycleManager)
            try await jobQuiescenceMonitor.startQuiescenceMonitor()

            try await self.lifecycleManager.managed {
                let identityManager = IdentityManager(
                    useSelfSignedCert: self.config.grpc?.useSelfSignedCertificate == true,
                    metricsSystem: self.metricsSystem,
                    metricProcess: "cloudboardd"
                )
                if !self.insecureListener, identityManager.identity == nil {
                    CloudBoardDaemon.logger.error("Unable to load TLS identity, exiting.")
                    throw IdentityManagerError.unableToRunSecureService
                }

                if hotProperties != nil {
                    CloudBoardDaemon.logger.info("Hot properties are enabled.")
                } else {
                    CloudBoardDaemon.logger.info("Hot properties are disabled.")
                }

                let heartbeatPublisher = self.heartbeatPublisher
                if heartbeatPublisher != nil {
                    CloudBoardDaemon.logger.info("Heartbeats are enabled.")
                } else {
                    CloudBoardDaemon.logger.info("Heartbeats are disabled.")
                }
                await heartbeatPublisher?.updateCredentialProvider {
                    identityManager.identity?.credential
                }

                let serviceAddress = try await self.resolveServiceAddress()
                let serviceDiscovery: ServiceDiscoveryPublisher?
                if let injectedSD = self.serviceDiscovery {
                    CloudBoardDaemon.logger.info("Using service discovery injected for testing")
                    serviceDiscovery = injectedSD
                } else if let sdConfig = self.config.serviceDiscovery {
                    CloudBoardDaemon.logger.info("Enabling service discovery")
                    serviceDiscovery = try ServiceDiscoveryPublisher(
                        group: self.group,
                        configuration: sdConfig,
                        serviceAddress: serviceAddress,
                        localIdentityCallback: identityManager.identityCallback,
                        hotProperties: hotProperties,
                        nodeInfo: nodeInfo,
                        cellID: self.config.serviceDiscovery?.cellID,
                        statusMonitor: self.statusMonitor,
                        metrics: self.metricsSystem
                    )
                } else {
                    CloudBoardDaemon.logger.warning("Service discovery not enabled")
                    serviceDiscovery = nil
                }

                let healthProvider = HealthProvider(monitor: self.healthMonitor)
                let healthServer = self.healthServer
                let attestationProvider: AttestationProvider
                if let injectedAttestationProvider = self.attestationProvider {
                    CloudBoardDaemon.logger.info("using attestation provider injected for testing")
                    attestationProvider = injectedAttestationProvider
                } else {
                    attestationProvider = await AttestationProvider(
                        attestationClient: CloudBoardAttestationAPIXPCClient.localConnection(),
                        metricsSystem: self.metricsSystem
                    )
                }

                let cloudboardProvider = CloudBoardProvider(
                    jobHelperClientProvider: self.jobHelperClientProvider,
                    jobHelperResponseDelegateProvider: self.jobHelperResponseDelegateProvider,
                    healthMonitor: self.healthMonitor,
                    metrics: self.metricsSystem,
                    tracer: self.tracer,
                    attestationProvider: attestationProvider,
                    loadConfiguration: self.config.load,
                    hotProperties: hotProperties
                )
                cloudBoardProvider.withLock { $0 = cloudboardProvider }

                let expectedPeerAPRN = self.config.grpc?.expectedPeerAPRN
                let keepalive = self.config.grpc?.keepalive
                let identityCallback = if !self.insecureListener {
                    identityManager.identityCallback
                } else {
                    nil as GRPCTLSConfiguration.IdentityCallback?
                }

                try await withErrorLogging(operation: "cloudboardDaemon task group", sensitiveError: false) {
                    try await withThrowingTaskGroup(of: Void.self) { group in
                        if let requestFielderManager = self.requestFielderManager {
                            group.addTask {
                                try await requestFielderManager.run()
                            }
                        }

                        group.addTask {
                            try await attestationProvider.run()
                        }

                        // Block service announce until we are sure we are able to fetch attestations
                        try await withErrorLogging(operation: "Verify attestation fetch", level: .default) {
                            self.statusMonitor.waitingForFirstAttestationFetch()
                            _ = try await attestationProvider.currentAttestationSet()
                        }

                        // Block service announce until we have at least one set of signing keys in cb_jobauthd
                        try await withErrorLogging(
                            operation: "Verify presence of auth token signing keys",
                            level: .default
                        ) {
                            if self.config.blockHealthinessOnAuthSigningKeysPresence {
                                self.statusMonitor.waitingForFirstKeyFetch()
                                try await self.checkAuthTokenSigningKeysPresence()
                                CloudBoardDaemon.logger.log("Signing key verification passed")
                            } else {
                                Self.logger
                                    .log(
                                        "Skipping verification due to config's blockHealthinessOnAuthSigningKeysPresence"
                                    )
                            }
                        }
                        self.statusMonitor.waitingForFirstHotPropertyUpdate()

                        group.addTask {
                            await cloudboardProvider.run()
                        }

                        group.addTaskWithLogging(operation: "certificate refresh", sensitiveError: false) {
                            await identityManager.identityUpdateLoop()
                        }

                        if let serviceDiscovery {
                            group.addTaskWithLogging(operation: "serviceDiscovery", sensitiveError: false) {
                                try await hotProperties?.waitForFirstUpdate()
                                self.statusMonitor.waitingForWorkloadRegistration()
                                do {
                                    try await serviceDiscovery.run()
                                } catch let error as CancellationError {
                                    throw error
                                } catch {
                                    self.statusMonitor.serviceDiscoveryRunningFailed()
                                    throw error
                                }
                            }
                        }

                        group.addTaskWithLogging(operation: "healthMonitor", sensitiveError: false) {
                            try await hotProperties?.waitForFirstUpdate()
                            await withLifecycleManagementHandlers(label: "healthMonitor") {
                                await healthProvider.run()
                            } onDrain: {
                                self.healthMonitor.drain()
                            }
                        }

                        if let healthServer {
                            group.addTaskWithLogging(operation: "healthServer", sensitiveError: false) {
                                await healthServer.run(healthPublisher: self.healthMonitor)
                            }
                        }

                        if let heartbeatPublisher {
                            group.addTaskWithLogging(operation: "heartbeat", sensitiveError: false) {
                                try await heartbeatPublisher.run()
                            }
                        }

                        group.addTaskWithLogging(operation: "gRPC server", sensitiveError: false) {
                            do {
                                try await Self.runServer(
                                    cloudBoardProvider: cloudboardProvider,
                                    providers: [healthProvider],
                                    identityCallback: identityCallback,
                                    serviceAddress: serviceAddress,
                                    expectedPeerAPRN: expectedPeerAPRN.map { try APRN(string: $0) },
                                    keepalive: keepalive.map { .init($0) },
                                    watchdogService: self.watchdogService,
                                    portPromise: portPromise,
                                    metricsSystem: self.metricsSystem
                                )
                            } catch {
                                self.statusMonitor.grpcServerRunningFailed()
                                throw error
                            }
                        }

                        if let workloadController = self.workloadController {
                            group.addTaskWithLogging(operation: "workloadController", sensitiveError: false) {
                                try await withLifecycleManagementHandlers(label: "workloadController") {
                                    do {
                                        try await workloadController.run(
                                            serviceDiscoveryPublisher: serviceDiscovery,
                                            concurrentRequestCountStream: cloudboardProvider
                                                .concurrentRequestCountStream,
                                            providerPause: cloudboardProvider.pause
                                        )

                                    } catch {
                                        self.statusMonitor.workloadControllerRunningFailed()
                                        throw error
                                    }
                                } onDrain: {
                                    do {
                                        try await workloadController.shutdown()
                                    } catch {
                                        Self.logger
                                            .error(
                                                "workload controller failed to notify listeners of shutdown: \(String(reportable: error), privacy: .public)"
                                            )
                                    }
                                }
                            }
                        }

                        if let hotProperties = self.hotProperties {
                            group.addTaskWithLogging(operation: "Hot properties task", sensitiveError: false) {
                                try await hotProperties.run(metrics: self.metricsSystem)
                            }
                        }

                        // When any of these tasks exit, they all do.
                        _ = try await group.next()
                        group.cancelAll()
                    }
                }
            } onDrain: {
                drainTimeMeasurement.withLock { $0 = ContinuousTimeMeasurement.start() }
            } onDrainCompleted: {
                let activeRequests = cloudBoardProvider.withLock { $0?.activeRequestsBeforeDrain }
                let drainDuration = drainTimeMeasurement.withLock { $0?.duration }
                if let activeRequests, let drainDuration {
                    self.metricsSystem.emit(
                        Metrics.CloudBoardDaemon.DrainCompletionTimeHistogram(
                            duration: drainDuration,
                            activeRequests: activeRequests
                        )
                    )
                    Self.logger.log("Drain Completed in: \(drainDuration.seconds)s")
                }
            }
        } catch {
            self.statusMonitor.daemonExitingOnError()
            CloudBoardDaemon.logger.error("fatal error, exiting: \(String(unredacted: error), privacy: .public)")
            throw error
        }

        self.statusMonitor.daemonDrained()
        await jobQuiescenceMonitor.quiesceCompleted()

        // once drained do not exit, JobQuiescence Framework will take care of exiting
        // stash continuation for use in tests where we may want to exit
        await withCheckedContinuation { exitContinuation in
            self.exitContinuation = exitContinuation
            if self._allowExit {
                exitContinuation.resume()
            }
        }
    }

    private var _allowExit = false

    private func checkAuthTokenSigningKeysPresence() async throws {
        let jobAuthClient: CloudBoardJobAuthAPIClientProtocol = if let _ = self.jobAuthClient {
            self.jobAuthClient!
        } else {
            await CloudBoardJobAuthAPIXPCClient.localConnection()
        }
        await jobAuthClient.connect()
        let tgtSigningPublicKeyDERs = try await jobAuthClient.requestTGTSigningKeys()
        let ottSigningPublicKeyDERs = try await jobAuthClient.requestOTTSigningKeys()

        guard !tgtSigningPublicKeyDERs.isEmpty, !ottSigningPublicKeyDERs.isEmpty else {
            throw InitializationError.missingAuthTokenSigningKeys
        }
    }

    private static func runServer(
        cloudBoardProvider: CloudBoardProvider,
        providers: [CallHandlerProvider],
        identityCallback: GRPCTLSConfiguration.IdentityCallback?,
        serviceAddress: SocketAddress,
        expectedPeerAPRN: APRN?,
        keepalive: ServerConnectionKeepalive?,
        watchdogService: CloudBoardWatchdogService?,
        portPromise: Promise<Int, Error>? = nil,
        metricsSystem: MetricsSystem
    ) async throws {
        // register watchdog work processor to monitor global concurrency thread pool.
        // Disabled in unit tests
        if let watchdogService {
            await watchdogService.activate()
        }

        let group = NIOTSEventLoopGroup(loopCount: 1)
        defer {
            try? group.syncShutdownGracefully()
        }

        let server = try await self.runGRPCServer(
            group: group,
            providers: [cloudBoardProvider] + providers,
            identityCallback: identityCallback,
            serviceAddress: serviceAddress,
            expectedPeerAPRN: expectedPeerAPRN,
            keepalive: keepalive,
            portPromise: portPromise,
            metricsSystem: metricsSystem
        )

        try await withLifecycleManagementHandlers(label: "gRPC server") {
            try await withTaskCancellationHandler {
                try await server.onClose.get()
            } onCancel: {
                server.close(promise: nil)
            }
        } onDrain: {
            await cloudBoardProvider.drain()
        }
    }

    private static func runGRPCServer(
        group: NIOTSEventLoopGroup,
        providers: [CallHandlerProvider],
        identityCallback: GRPCTLSConfiguration.IdentityCallback?,
        serviceAddress: SocketAddress,
        expectedPeerAPRN: APRN?,
        keepalive: ServerConnectionKeepalive?,
        portPromise: Promise<Int, Error>? = nil,
        metricsSystem: MetricsSystem
    ) async throws -> Server {
        let loggingLogger = Logging.Logger(
            osLogSubsystem: "com.apple.cloudos.cloudboard",
            osLogCategory: "cloudboardd",
            domain: "GRPCServer"
        )

        let server: Server

        CloudBoardDaemon.logger.log("Running GRPC service at \(serviceAddress, privacy: .public)")

        do {
            if let identityCallback {
                CloudBoardDaemon.logger.info("Running service with TLS.")
                let config = try GRPCTLSConfiguration.cloudboardProviderConfiguration(
                    identityCallback: identityCallback,
                    expectedPeerAPRN: expectedPeerAPRN,
                    metricsSystem: metricsSystem
                )
                var serverBuilder = Server.usingTLS(with: config, on: group)
                    .withServiceProviders(providers)
                    .withLogger(loggingLogger)
                if let keepalive {
                    CloudBoardDaemon.logger.info("Configuring GRPCServer keepalive")
                    serverBuilder = serverBuilder
                        .withKeepalive(keepalive)
                }
                server = try await serverBuilder.bind(host: serviceAddress.ipAddress!, port: serviceAddress.port!).get()
            } else {
                CloudBoardDaemon.logger.warning("Running service without TLS.")
                server = try await Server.insecure(group: group)
                    .withServiceProviders(providers)
                    .withLogger(loggingLogger)
                    .bind(host: serviceAddress.ipAddress!, port: serviceAddress.port!).get()
            }
        } catch {
            portPromise?.fail(with: error)
            throw error
        }

        portPromise?.succeed(with: server.channel.localAddress!.port!)

        CloudBoardDaemon.logger
            .info("Bound service at \(String(describing: server.channel.localAddress), privacy: .public)")

        return server
    }

    private func resolveServiceAddress() async throws -> SocketAddress {
        return try await self.config.resolveLocalServiceAddress()
    }
}

extension UUID {
    static let zero = UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
}

// NOTE: The description of this type is publicly logged and/or included in metric dimensions and therefore MUST not
// contain sensitive data.
struct CloudBoardDaemonLogMetadata: CustomStringConvertible {
    var jobID: UUID?
    var rpcID: UUID?
    var requestTrackingID: String?
    var remotePID: Int?

    init(jobID: UUID? = nil, rpcID: UUID? = nil, requestTrackingID: String? = nil, remotePID: Int? = nil) {
        self.jobID = jobID
        self.rpcID = rpcID
        self.requestTrackingID = requestTrackingID
        self.remotePID = remotePID
    }

    var description: String {
        var text = ""

        text.append("[")
        if let jobID = self.jobID {
            text.append("jobID=\(jobID) ")
        }
        if let rpcID = self.rpcID, rpcID != .zero {
            text.append("rpcID=\(rpcID) ")
        }
        if let requestTrackingID = self.requestTrackingID, requestTrackingID != "" {
            text.append("requestTrackingID=\(requestTrackingID) ")
        }
        if let remotePID = self.remotePID {
            text.append("remotePID=\(remotePID) ")
        }

        text.removeLast(1)
        if !text.isEmpty {
            text.append("]")
        }

        return text
    }
}

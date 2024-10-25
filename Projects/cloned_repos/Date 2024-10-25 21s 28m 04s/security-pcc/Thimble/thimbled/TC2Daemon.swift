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
//  TC2Daemon.swift
//  PrivateCloudCompute
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import CloudTelemetry
import Foundation
import Foundation_Private.NSBackgroundActivityScheduler
import Foundation_Private.NSXPCConnection
import PrivateCloudCompute
import os.lock

@available(iOS 18.0, *)
@main
final class TC2Daemon: NSObject, NSXPCListenerDelegate, TC2DaemonHostDelegate, Sendable, TC2Sandbox {
    let logger = tc2Logger(forCategory: .Daemon)
    let serverDrivenConfig: TC2ServerDrivenConfiguration
    let config = TC2DefaultConfiguration()
    let prefetchActivity: TC2PrefetchActivity
    let updateServerDrivenConfigurationActivity: TC2UpdateServerDrivenConfigurationActivity
    let rateLimiter: RateLimiter
    let attestationStore: TC2AttestationStore?
    let attestationVerifier: TC2CloudAttestationVerifier
    let dailyActiveUserReporter = DailyActiveUsersReporter<UserDefaultsStore>()
    let tapToRadarController = TapToRadarController()
    let nodeDistributionAnalyzer: NodeDistributionAnalyzer
    let nodeDistributionReportActivity: NodeDistributionAnalyzerReportActivity

    let (thimbledEventStream, thimbledEventContinuation) = AsyncStream.makeStream(of: ThimbledEvent.self)

    var scheduledPrefetchTask: TC2ScheduledTask {
        return .init(preregisteredIdentifier: "com.apple.privatecloudcompute.prefetchAttestations", work: prefetchActivity)
    }

    var scheduledFetchConfigBagTask: TC2ScheduledTask {
        return .init(preregisteredIdentifier: "com.apple.privatecloudcompute.fetchServerDrivenConfig", work: updateServerDrivenConfigurationActivity)
    }

    var scheduledNodeDistributionReportTask: TC2ScheduledTask {
        return .init(preregisteredIdentifier: "com.apple.privatecloudcompute.nodeDistributionReport", work: nodeDistributionReportActivity)
    }

    let structuredRequestFactoriesBySetup: OSAllocatedUnfairLock<[TC2ResolvedSetup: ThimbledTrustedRequestFactory]> = OSAllocatedUnfairLock(initialState: [:])

    var requestMetadata: TC2TrustedRequestFactoriesMetadata {
        let metadata = self.structuredRequestFactoriesBySetup.withLock { $0.values.compactMap { $0.getRequestMetadata() } }
        return TC2TrustedRequestFactoriesMetadata(requests: metadata)
    }

    let prefetchTracker = TC2PrefetchTracker()

    static func main() async {
        let instance = TC2Daemon()
        await instance.main()
    }

    override init() {
        let daemonDirectoryPath = getDaemonDirectoryPath()
        let environment = self.config.environment
        self.rateLimiter = RateLimiter(config: self.config, from: daemonDirectoryPath)
        self.attestationStore = TC2AttestationStore(environment: environment, dir: daemonDirectoryPath)
        self.serverDrivenConfig = TC2ServerDrivenConfiguration(from: daemonDirectoryPath)
        self.attestationVerifier = TC2CloudAttestationVerifier(environment: environment)
        self.prefetchActivity = TC2PrefetchActivity(
            rateLimiter: self.rateLimiter,
            attestationStore: self.attestationStore,
            attestationVerifer: self.attestationVerifier,
            config: self.config,
            daemonDirectoryPath: daemonDirectoryPath,
            serverDrivenConfig: self.serverDrivenConfig,
            eventStreamContinuation: self.thimbledEventContinuation,
            prefetchTracker: self.prefetchTracker
        )
        self.updateServerDrivenConfigurationActivity = TC2UpdateServerDrivenConfigurationActivity(
            serverDrivenConfig: self.serverDrivenConfig,
            config: self.config
        )
        self.nodeDistributionAnalyzer = NodeDistributionAnalyzer(
            environment: environment.name,
            storeURL: daemonDirectoryPath
        )
        self.nodeDistributionReportActivity = NodeDistributionAnalyzerReportActivity(
            eventStreamContinuation: self.thimbledEventContinuation,
            nodeDistributionAnalyzer: self.nodeDistributionAnalyzer
        )

        super.init()
        logger.log("Starting daemon. tc2OSInfo: \(tc2OSInfo)")
    }

    func structuredRequestFactory(forSetup setup: TC2ResolvedSetup) -> ThimbledTrustedRequestFactory {
        self.structuredRequestFactoriesBySetup.withLock { factories in
            if let factory = factories[setup] {
                return factory
            } else {
                let factory = ThimbledTrustedRequestFactory(
                    config: self.config,
                    serverDrivenConfig: self.serverDrivenConfig,
                    connectionFactory: NWAsyncConnection(),
                    attestationStore: self.attestationStore,
                    attestationVerifier: self.attestationVerifier,
                    rateLimiter: self.rateLimiter,
                    tokenProvider: TC2NSPTokenProvider(config: self.config),
                    clock: ContinuousClock(),
                    clientBundleIdentifier: setup.clientBundleIdentifier,
                    allowBundleIdentifierOverride: setup.allowBundleIdentifierOverride,
                    parametersCache: self.prefetchActivity.parametersCache,
                    eventStreamContinuation: self.thimbledEventContinuation
                )
                factories[setup] = factory
                return factory
            }
        }
    }

    func main() async {
        self.logger.log("Entering sandbox")
        Self.enterSandbox(identifier: "com.apple.privatecloudcomputed", macOSProfile: "com.apple.privatecloudcomputed")
        do {
            self.logger.log("Setting up CloudTelemetry xpc service activities.")
            try await CloudTelemetry.setupXpcServiceActivities()
        } catch {
            self.logger.log("Failed to setup CloudTelemetry")
        }

        DispatchQueue.global().async {
            let listener = NSXPCListener(machServiceName: "com.apple.privatecloudcompute")
            listener.delegate = self

            self.logger.log("Listener start")
            listener.resume()
            self.logger.log("Listener done")

            self.logger.log("Register prefetch task")
            self.scheduledPrefetchTask.register()

            self.logger.log("Register fetch config bag task")
            self.scheduledFetchConfigBagTask.register()

            self.logger.log("Register node distribution report task")
            self.scheduledNodeDistributionReportTask.register()
        }

        logger.log("Starting daemon run loop...")

        let metricExporter = TC2MetricReporter()

        // thimbled event stream should never be finished
        await withDiscardingTaskGroup { taskGroup in
            for await event in self.thimbledEventStream {
                switch event {
                case .expiredAttestationList(let expiredAttestationList, let parameters):
                    taskGroup.addTask {
                        await withOSTransaction(name: "com.apple.privatecloudcomputed.prefetch") {
                            await self.prefetchAttestationsAsResponseToExpiredAttestationList(
                                expiredAttestationList,
                                parameters: parameters
                            )
                        }
                    }

                case .reportDailyActiveUserIfNecessary(let requestID, let environment):
                    taskGroup.addTask {
                        if let metric = await self.dailyActiveUserReporter.makeDailyActiveUserReportEvent(requestID: requestID, environment: environment) {
                            await metricExporter.reportCloudTelemetryMetric(metric: metric)
                        }
                    }

                case .exportMetric(let metric):
                    taskGroup.addTask {
                        await metricExporter.reportCloudTelemetryMetric(metric: metric)
                    }

                case .rateLimitConfigurations(let rateLimitConfigs):
                    taskGroup.addTask {
                        for rateLimitConfig in rateLimitConfigs {
                            await self.rateLimiter.limitByConfiguration(rateLimitConfig)
                        }
                        await self.rateLimiter.trimExpiredData()
                        await self.rateLimiter.save()
                    }

                case .attestationStoreCleanup:
                    taskGroup.addTask {
                        await self.attestationStore?.deleteEntriesWithExpiredAttestationBundles()
                    }

                case .prewarmAttestations(let workloadType, let workloadParameters, let bundleIdentifier, let featureIdentifier):
                    taskGroup.addTask {
                        self.logger.log("running prewarmAttestations: \(workloadType) \(workloadParameters)")
                        await withOSTransaction(name: "com.apple.privatecloudcomputed.prewarm") {
                            _ = await self.handlePrefetchRequest(
                                workloadType: workloadType,
                                workloadParameters: workloadParameters,
                                prewarm: true,
                                fetchType: .fetchAllBatches,
                                bundleIdentifier: bundleIdentifier,
                                featureIdentifier: featureIdentifier
                            )
                        }
                    }

                case .prefetchAttestationsForNewWorkload(let parameters):
                    taskGroup.addTask {
                        self.logger.log("running prefetchAttestationsForNewWorkload")
                        await withOSTransaction(name: "com.apple.privatecloudcomputed.prefetch") {
                            _ = await self.handlePrefetchRequest(
                                workloadType: parameters.pipelineKind,
                                workloadParameters: parameters.pipelineArguments,
                                prewarm: false,
                                fetchType: .fetchAllBatches,
                                bundleIdentifier: nil,
                                featureIdentifier: nil
                            )
                        }
                    }

                case .discardUsedAttestationsAndPrefetchBatch(let serverRequestID, let parameters):
                    taskGroup.addTask {
                        self.logger.log("running discardUsedAttestationsAndPrefetchBatch")
                        guard let attestationStore = self.attestationStore else {
                            self.logger.error("failed to prefetch attestations as store is not initialized")
                            return
                        }

                        await withOSTransaction(name: "com.apple.privatecloudcomputed.prefetch") {
                            // 1. Issue a delete request to the store with trustedRequestID
                            let batchID = await attestationStore.deleteAttestationsUsedByTrustedRequest(serverRequestID: serverRequestID)

                            // 2. Prefetch a single batch to top up the store
                            _ = await self.handlePrefetchRequest(
                                workloadType: parameters.pipelineKind,
                                workloadParameters: parameters.pipelineArguments,
                                prewarm: false,
                                fetchType: .fetchSingleBatch(batchID: batchID),
                                bundleIdentifier: nil,
                                featureIdentifier: nil
                            )
                        }
                    }

                case .nodesReceived(let nodeIDs, let source):
                    taskGroup.addTask {
                        await self.nodeDistributionAnalyzer.receivedNodesWith(nodeIDs: nodeIDs, from: source)
                    }
                case .tapToRadarIndicationReceived(let context):
                    taskGroup.addTask {
                        _ = await self.tapToRadarController.ttrIndicationReceived(context)
                    }
                }
            }
        }
    }

    /// This method is where the NSXPCListener configures, accepts, and resumes a new incoming NSXPCConnection.
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        logger.log("Thimble trying to connect: checking entitlements")

        let hasEntitlement = TC2Entitlement.allCases.contains { entitlement in
            return newConnection.value(forEntitlement: entitlement.rawValue) as? Bool == true
        }
        guard hasEntitlement else {
            logger.log("Rejecting connection because it doesn't have any of the required entitlements: \(String(describing: TC2Entitlement.allCases))")
            return false
        }

        // set up the new connection with an exported ThimbleDaemonHandler

        // Configure the connection.
        // First, set the interface that the exported object implements.
        newConnection.exportedInterface = interfaceForTC2DaemonProtocol()

        // Next, set the object that the connection exports. All messages sent on the connection to this service will be sent to the exported object to handle. The connection retains the exported object.
        let exportedObject = TC2DaemonHost(config: self.config, delegate: self, connection: newConnection)
        newConnection.exportedObject = exportedObject

        // Resuming the connection allows the system to deliver more incoming messages.
        newConnection.resume()
        logger.log("Resumed")
        newConnection.invalidationHandler = { [logger] in
            logger.log("XPC connection invalidated")
        }
        newConnection.interruptionHandler = { [logger] in
            logger.log("XPC connection interrupted")
        }

        // Returning true from this method tells the system that you have accepted this connection. If you want to reject the connection for some reason, call invalidate() on the connection and return false.
        return true
    }

    func prefetchAttestationsAsResponseToExpiredAttestationList(
        _ expiredAttestationList: Proto_Ropes_HttpService_InvokeResponse.ExpiredAttestationList,
        parameters: TC2RequestParameters
    ) async {
        do {
            if expiredAttestationList.shouldClearCache {
                if let attestationStore = self.attestationStore {
                    // The prefetch request will take care of clearing out the store of older attestations for this workload
                    self.logger.log("creating a prefetch request as response to expired attestations")
                    try await self.prefetchTracker.ensuringOnlyASinglePrefetchIsRunningForParameters(parameters) {
                        let prefetchRequest = TC2BatchedPrefetch(
                            connectionFactory: NWAsyncConnection(),
                            attestationStore: attestationStore,
                            rateLimiter: self.rateLimiter,
                            attestationVerifier: self.attestationVerifier,
                            config: self.config,
                            serverDrivenConfig: self.serverDrivenConfig,
                            parameters: .init(
                                pipelineKind: parameters.pipelineKind,
                                pipelineArguments: parameters.pipelineArguments
                            ),
                            eventStreamContinuation: self.thimbledEventContinuation,
                            prewarm: false,
                            fetchType: .fetchAllBatches
                        )

                        self.logger.log("firing prefetch request as response to expired attestations")
                        _ = try await prefetchRequest.sendRequest()
                        self.logger.log("succeeded prefetch request as response to expired attestations")
                    }
                }
            } else {
                for nodeIdentifier in expiredAttestationList.nodeIdentifier {
                    _ = await self.attestationStore?.deleteEntryForNode(nodeIdentifier: nodeIdentifier)
                }
            }
        } catch {
            self.logger.log("failed prefetch request as response to expired attestations. error: \(error)")
        }
    }
}

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

import AppServerSupport.OSLaunchdJob
import CloudBoardAsyncXPC
import CloudBoardAttestationDAPI
import CloudBoardCommon
import CloudBoardJobAPI
import CloudBoardJobAuthDAPI
import CloudBoardJobHelperAPI
import CloudBoardMetrics
import CloudBoardPreferences
import Foundation
import os

enum CloudBoardJobHelperError: Error {
    case unableToFindCloudAppToManage
}

struct CloudBoardJobHelperHotProperties: Decodable, Hashable {
    private enum CodingKeys: String, CodingKey {
        case _maxRequestMessageSize = "MaxRequestMessageSize"
    }

    private var _maxRequestMessageSize: Int?
    var maxRequestMessageSize: Int {
        self._maxRequestMessageSize ?? 1024 * 1024 * 4 // 4MB
    }
}

/// Per-request process implementing an end-to-end encrypted protocol with privatecloudcomputed on the client. It is
/// responsible for decrypting the request stream and encrypting the response stream to and from the cloud app.
/// cb_jobhelper also implements authentication of client requests by verifying the signature of the TGT sent by
/// privatecloudcomputed with the request as well as verifying that the OTT provided by the PCC Gateway is derived from
/// the TGT.
public actor CloudBoardJobHelper {
    static let metricsClientName = "cb_jobhelper"

    let server: CloudBoardJobHelperAPIServerProtocol
    let attestationClient: CloudBoardAttestationAPIClientProtocol?
    let jobAuthClient: CloudBoardJobAuthAPIClientProtocol?
    let metrics: any MetricsSystem
    var requestID: String
    var jobUUID: UUID

    public static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "cb_jobhelper"
    )

    public init() {
        self.server = CloudBoardJobHelperAPIXPCServer.localListener()
        self.attestationClient = nil
        self.jobAuthClient = nil
        self.metrics = CloudMetricsSystem(clientName: Self.metricsClientName)
        self.requestID = ""

        let myUUID = LaunchdJobHelper.currentJobUUID(logger: Self.logger)
        if myUUID == nil {
            Self.logger.warning("Could not get own job UUID, creating new UUID for app")
        }
        self.jobUUID = myUUID ?? UUID()
    }

    // For testing only
    internal init(
        server: CloudBoardJobHelperAPIServerProtocol,
        attestationClient: CloudBoardAttestationAPIClientProtocol,
        jobAuthClient: CloudBoardJobAuthAPIClientProtocol,
        metrics: any MetricsSystem
    ) {
        self.server = server
        self.attestationClient = attestationClient
        self.jobAuthClient = jobAuthClient
        self.metrics = metrics
        self.requestID = ""

        if let currentUUID = LaunchdJobHelper.currentJobUUID(logger: Self.logger) {
            self.jobUUID = currentUUID
        } else {
            let jobUUID = UUID()
            self.jobUUID = jobUUID
            Self.logger.warning("""
            jobID=\(jobUUID.uuidString, privacy: .public)
            message=\("Could not get own job UUID, creating new UUID for app")
            """)
        }

        /// Workaround to prevent Swift compiler from ignoring all conformances defined in Logging+ReportableError
        /// rdar://126351696 (Swift compiler seems to ignore protocol conformances not used in the same target)
        _ = CloudBoardJobHelperError.unableToFindCloudAppToManage.publicDescription
    }

    /// Emit the launch metrics.
    private func emitLaunchMetrics() {
        self.metrics.emit(Metrics.Daemon.LaunchCounter(action: .increment))
    }

    /// Updates the metrics with the current values, called periodically.
    /// - Parameter startInstant: The instant at which the daemon started.
    private func updateMetrics(startInstant: ContinuousClock.Instant) {
        let uptime = Int(clamping: startInstant.duration(to: .now).components.seconds)
        self.metrics.emit(Metrics.Daemon.UptimeGauge(value: uptime))
    }

    private func setRequestID(_ requestID: String) {
        self.requestID = requestID
    }

    public func start() async throws {
        CloudboardJobHelperCheckpoint(logMetadata: logMetadata(), message: "Starting")
            .log(to: Self.logger, level: .default)
        defer {
            self.metrics.emit(Metrics.Daemon.TotalExitCounter(action: .increment))
            self.metrics.invalidate()
            CloudboardJobHelperCheckpoint(logMetadata: logMetadata(), message: "Finished")
                .log(to: Self.logger, level: .default)
        }
        self.emitLaunchMetrics()

        let preferencesUpdates = PreferencesUpdates(
            preferencesDomain: "com.apple.cloudos.hotproperties.cb_jobhelper",
            maximumUpdateDuration: .seconds(1),
            forType: CloudBoardJobHelperHotProperties.self
        )

        // force unwrap is safe as we will either get the preferences or throw an error
        let preferences = try await preferencesUpdates.first(where: { _ in true })!.applyingPreferences { $0 }

        let config = try CBJobHelperConfiguration.fromPreferences()

        // Create streams for communication between the different cb_jobhelper components
        let (wrappedRequestStream, wrappedRequestContinuation) = AsyncStream<PipelinePayload<Data>>.makeStream()
        let (wrappedResponseStream, wrappedResponseContinuation) = AsyncStream<FinalizableChunk<Data>>.makeStream()
        let (cloudAppRequestStream, cloudAppRequestContinuation) = AsyncStream<PipelinePayload<Data>>.makeStream()
        let (cloudAppResponseStream, cloudAppResponseContinuation) = AsyncThrowingStream<CloudAppResponse, Error>
            .makeStream()

        // Fetch signing keys for TGT and OTT signature verification and register for updates
        let jobAuthClient: CloudBoardJobAuthAPIClientProtocol = if self.jobAuthClient != nil {
            self.jobAuthClient!
        } else {
            await CloudBoardJobAuthAPIXPCClient.localConnection()
        }
        await jobAuthClient.connect()

        let signingKeySet: AuthTokenKeySet
        do {
            signingKeySet = try await .init(
                ottPublicSigningKeys: jobAuthClient.requestOTTSigningKeys(),
                tgtPublicSigningKeys: jobAuthClient.requestTGTSigningKeys()
            )
        } catch {
            if config.enforceTGTValidation {
                CloudboardJobHelperCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Could not load signing keys from cb_jobauthd. Failing.",
                    error: error
                ).log(to: Self.logger, level: .fault)
                throw error
            } else {
                CloudboardJobHelperCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Could not load signing keys from cb_jobauthd. Continuing.",
                    error: error
                ).log(to: Self.logger, level: .fault)
                signingKeySet = .init(ottPublicSigningKeys: [], tgtPublicSigningKeys: [])
            }
        }

        let cloudBoardMessenger = CloudBoardMessenger(
            attestationClient: self.attestationClient,
            server: self.server,
            encodedRequestContinuation: wrappedRequestContinuation,
            responseStream: wrappedResponseStream,
            metrics: self.metrics
        )
        let tgtValidator = TokenGrantingTokenValidator(signingKeys: signingKeySet)

        let jobAuthClientDelegate = JobAuthClientDelegate(tgtValidator: tgtValidator)
        await jobAuthClient.set(delegate: jobAuthClientDelegate)

        let workloadJobManager = WorkloadJobManager(
            tgtValidator: tgtValidator,
            enforceTGTValidation: config.enforceTGTValidation,
            requestStream: wrappedRequestStream,
            maxRequestMessageSize: preferences.maxRequestMessageSize,
            responseContinuation: wrappedResponseContinuation,
            cloudAppRequestContinuation: cloudAppRequestContinuation,
            cloudAppResponseStream: cloudAppResponseStream,
            cloudAppResponseContinuation: cloudAppResponseContinuation,
            metrics: self.metrics,
            jobUUID: self.jobUUID
        )

        let cloudAppResponseDelegate = CloudAppResponseDelegate(
            responseContinuation: cloudAppResponseContinuation
        )

        let workload = try getCloudAppWorkload(
            cloudAppNameOverride: config.cloudAppName,
            delegate: cloudAppResponseDelegate,
            jobUUID: self.jobUUID
        )

        await self.server.set(delegate: cloudBoardMessenger)
        await self.server.connect()

        do {
            try await self.withMetricsUpdates {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    do {
                        group.addTaskWithLogging(
                            operation: "cloudBoardMessenger.run()",
                            metrics: .init(
                                metricsSystem: self.metrics,
                                errorFactory: Metrics.Messenger.OverallErrorCounter.Factory()
                            ),
                            logger: Self.logger
                        ) {
                            try await cloudBoardMessenger.run()
                        }
                        group.addTaskWithLogging(
                            operation: "workloadJobManager.run()",
                            metrics: .init(
                                metricsSystem: self.metrics,
                                errorFactory: Metrics.WorkloadManager.OverallErrorCounter.Factory()
                            ),
                            logger: Self.logger
                        ) {
                            await workloadJobManager.run()
                        }
                        group.addTaskWithLogging(
                            operation: "workload.run()",
                            metrics: .init(
                                metricsSystem: self.metrics,
                                errorFactory: Metrics.Workload.OverallErrorCounter.Factory()
                            ),
                            logger: Self.logger
                        ) { try await workload.run() }
                        group.addTaskWithLogging(
                            operation: "cloudAppRequestStream consumer",
                            metrics: .init(
                                metricsSystem: self.metrics,
                                errorFactory: Metrics.CloudAppRequestStream.OverallErrorCounter.Factory()
                            ),
                            logger: Self.logger
                        ) {
                            for try await request in cloudAppRequestStream {
                                switch request {
                                case .warmup(let warmupData):
                                    await CloudboardJobHelperCheckpoint(
                                        logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                                        message: "Forwarding warmup message to workload",
                                        operationName: "cloudAppRequestStream"
                                    ).log(to: Self.logger, level: .info)
                                    try await workload.warmup(warmupData)
                                case .parameters(let parametersData):
                                    await self.setRequestID(parametersData.plaintextMetadata.requestID)
                                    await CloudboardJobHelperCheckpoint(
                                        logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                                        message: "Forwarding parameters message to workload",
                                        operationName: "cloudAppRequestStream"
                                    ).log(to: Self.logger, level: .info)
                                    try await workload.parameters(parametersData)
                                case .chunk(let chunk):
                                    await CloudboardJobHelperCheckpoint(
                                        logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                                        message: "Forwarding chunk to workload",
                                        operationName: "cloudAppRequestStream"
                                    ).log(to: Self.logger, level: .info)
                                    try await workload.provideInput(chunk.chunk, isFinal: chunk.isFinal)
                                case .endOfInput:
                                    await CloudboardJobHelperCheckpoint(
                                        logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                                        message: "Forwarding end of input signal to workload",
                                        operationName: "cloudAppRequestStream"
                                    ).log(to: Self.logger, level: .info)
                                    try await workload.endOfInput()
                                case .teardown:
                                    await CloudboardJobHelperCheckpoint(
                                        logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                                        message: "Forwarding teardown message to workload",
                                        operationName: "cloudAppRequestStream"
                                    ).log(to: Self.logger, level: .info)
                                    try await workload.teardown()
                                case .oneTimeToken:
                                    // We don't forward one-time tokens to the cloud app and receiving any in the
                                    // cloudAppRequestStream is unexpected
                                    await CloudboardJobHelperCheckpoint(
                                        logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                                        message: "Unexpectedly received one-time token to be forwarded to the cloud app. Ignoring.",
                                        operationName: "cloudAppRequestStream"
                                    ).log(to: Self.logger, level: .fault)
                                }
                            }
                            await CloudboardJobHelperCheckpoint(
                                logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                                message: "Request stream finished",
                                operationName: "cloudAppRequestStream"
                            ).log(to: Self.logger, level: .default)
                            try await workload.provideInput(nil, isFinal: true)
                        }

                        try await group.waitForAll()
                        await CloudboardJobHelperCheckpoint(
                            logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                            message: "Completed cloud app request and response handling",
                            operationName: "cloudAppRequestStream"
                        ).log(to: Self.logger, level: .default)
                    } catch {
                        await CloudboardJobHelperCheckpoint(
                            logMetadata: self.logMetadata(withRemotePID: workload.remotePID),
                            message: "Error while managing cloud app. Attempting to tear down workload.",
                            error: error
                        ).log(to: Self.logger, level: .error)
                        // Attempt to cleanly tear down workload
                        // Teardown is idempotent, so it's fine to be called multiple times
                        try await workload.teardown()
                        // Rethrow error
                        throw error
                    }
                }
            }
        } catch {
            self.metrics.emit(Metrics.Daemon.ErrorExitCounter(action: .increment))
            CloudboardJobHelperCheckpoint(logMetadata: self.logMetadata(), message: "Finished", error: error)
                .log(to: Self.logger, level: .error)
        }

        withExtendedLifetime(jobAuthClientDelegate) {}
    }

    func withMetricsUpdates(body: @escaping @Sendable () async throws -> Void) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            let startInstant = ContinuousClock.now

            group.addTask {
                try await body()
            }

            group.addTaskWithLogging(operation: "Update metrics task", logger: Self.logger) {
                do {
                    while !Task.isCancelled {
                        await self.updateMetrics(startInstant: startInstant)
                        // only throws if the task is cancelled
                        try await Task.sleep(for: .seconds(5))
                    }
                } catch {
                    await CloudboardJobHelperCheckpoint(
                        logMetadata: self.logMetadata(),
                        message: "Update metric task cancelled",
                        error: error
                    ).log(to: Self.logger, level: .default)
                }
            }

            try await group.next()
            group.cancelAll()
        }
    }

    func getCloudAppWorkload(
        cloudAppNameOverride: String?,
        delegate: CloudBoardJobAPIClientDelegateProtocol,
        jobUUID: UUID
    ) throws -> CloudBoardJobHelperWorkload {
        var job: ManagedLaunchdJob?
        let managedJobs = LaunchdJobHelper.fetchManagedLaunchdJobs(
            type: CloudBoardJobType.cloudBoardApp,
            skippingInstances: true,
            logger: Self.logger
        )

        guard managedJobs.count >= 1 else {
            CloudboardJobHelperCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Failed to discover any managed CloudApps"
            ).log(to: Self.logger, level: .error)
            fatalError("Failed to discover any managed CloudApps")
        }

        let apps = if let cloudAppNameOverride {
            [cloudAppNameOverride]
        } else {
            ["TIECloudApp", "NullCloudApp"]
        }

        for app in apps {
            let managedJobsFiltered = managedJobs.filter { job in
                job.jobAttributes.cloudBoardJobName == app
            }
            if managedJobsFiltered.count == 0 {
                continue
            }
            guard managedJobsFiltered.count == 1 else {
                Self.logger.error(
                    "Found \(managedJobsFiltered.count, privacy: .public) instances of \(app, privacy: .public)"
                )
                fatalError()
            }
            Self.logger.log("Using app \(app, privacy: .public)")
            job = managedJobsFiltered[0]
            break
        }

        guard let job else {
            CloudboardJobHelperCheckpoint(
                logMetadata: self.logMetadata(),
                message: "No cloud app found"
            ).log(to: Self.logger, level: .error)
            fatalError("No cloud app found")
        }

        let workload = try CloudBoardAppWorkload(
            managedJob: job,
            machServiceName: job.jobAttributes.initMachServiceName,
            log: Self.logger,
            delegate: delegate,
            metrics: self.metrics,
            jobUUID: jobUUID
        ) as CloudBoardAppWorkload
        return workload
    }
}

// Delegate to handle output from the workload
private actor CloudAppResponseDelegate: CloudBoardJobAPIClientDelegateProtocol {
    public static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "cb_jobhelper.CloudAppResponseDelegate"
    )
    let responseContinuation: AsyncThrowingStream<CloudAppResponse, any Error>.Continuation
    init(responseContinuation: AsyncThrowingStream<CloudAppResponse, any Error>.Continuation) {
        self.responseContinuation = responseContinuation
    }

    func cloudBoardJobAPIClientSurpriseDisconnect() async {
        Self.logger.log("CloudAppResponseDelegate surpriseDisconnect")
        self.responseContinuation.finish()
    }

    func provideResponseChunk(_ data: Data) async throws {
        self.responseContinuation.yield(.chunk(data))
    }

    func endJob() async throws {
        // Might be called multiple times (see CloudBoardAppWorkload : monitoringCompleted)
        self.responseContinuation.finish()
    }

    func findHelper(helperID _: UUID) async throws {
        Self.logger.log("CloudAppResponseDelegate findHelper()")
        fatalError("Unimplemented")
    }

    func sendHelperMessage(helperID _: UUID, data _: Data) async throws {
        Self.logger.log("CloudAppResponseDelegate sendHelperMessage()")
        fatalError("Unimplemented")
    }

    func sendHelperEOF(helperID _: UUID) async throws {
        Self.logger.log("CloudAppResponseDelegate sendHelperEOF()")
        fatalError("Unimplemented")
    }

    func cloudBoardJobAPIClientAppTerminated(statusCode: Int?) async {
        self.responseContinuation.yield(.appTermination(.init(statusCode: statusCode)))
    }
}

// NOTE: The description of this type is publicly logged and/or included in metric dimensions and therefore MUST not
// contain sensitive data.
struct CloudBoardJobHelperLogMetadata: CustomStringConvertible {
    var jobID: UUID?
    var requestTrackingID: String?
    var remotePID: Int?

    init(jobID: UUID? = nil, requestTrackingID: String? = nil, remotePID: Int? = nil) {
        self.jobID = jobID
        self.requestTrackingID = requestTrackingID
        self.remotePID = remotePID
    }

    // NOTE: This description is publicly logged and/or included in metric dimensions and therefore MUST not contain
    // sensitive data.
    var description: String {
        var text = ""

        text.append("[")
        if let jobID = self.jobID {
            text.append("jobId=\(jobID) ")
        }
        if let requestTrackingID = self.requestTrackingID, requestTrackingID != "" {
            text.append("requestId=\(requestTrackingID) ")
        }
        if let remotePID = self.remotePID {
            text.append("remotePid=\(remotePID) ")
        }

        text.removeLast(1)
        if !text.isEmpty {
            text.append("]")
        }

        return text
    }
}

extension UUID {
    static let zero = UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
}

extension SecKey {
    static func fromDEREncodedRSAPublicKey(_ derEncodedRSAPublicKey: Data) throws -> SecKey {
        let keyAttributes: [CFString: Any] = [
            kSecAttrKeyType: kSecAttrKeyTypeRSA,
            kSecAttrKeyClass: kSecAttrKeyClassPublic,
        ]
        var error: Unmanaged<CFError>?
        let key = SecKeyCreateWithData(derEncodedRSAPublicKey as CFData, keyAttributes as CFDictionary, &error)

        guard let key else {
            // If this returns nil, error must be set.
            throw error!.takeRetainedValue() as Error
        }

        return key
    }
}

extension CloudBoardJobHelper {
    private func logMetadata(withRemotePID remotePID: Int? = nil) -> CloudBoardJobHelperLogMetadata {
        return CloudBoardJobHelperLogMetadata(
            requestTrackingID: self.requestID,
            remotePID: remotePID
        )
    }
}

// Delegate used to listen for TGT/OTT signing key updates
private actor JobAuthClientDelegate: CloudBoardJobAuthAPIClientDelegateProtocol {
    private let tgtValidator: TokenGrantingTokenValidator

    init(tgtValidator: TokenGrantingTokenValidator) {
        self.tgtValidator = tgtValidator
    }

    func surpriseDisconnect() async {
        CloudBoardJobHelper.logger.error("Surprise disconnect from cb_jobauthd")
    }

    func authKeysUpdated(newKeySet: AuthTokenKeySet) async throws {
        CloudBoardJobHelper.logger.log("Received new set of signing keys from cb_jobauthd")
        self.tgtValidator.setSigningKeys(newKeySet)
    }
}

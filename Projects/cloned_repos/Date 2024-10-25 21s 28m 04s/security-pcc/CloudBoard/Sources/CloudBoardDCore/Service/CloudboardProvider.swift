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
import CloudBoardJobHelperAPI
import CloudBoardLogging
import CloudBoardMetrics
import CloudBoardPlatformUtilities
import CryptoKit
import Foundation
import InternalGRPC
import InternalSwiftProtobuf
import NIOCore
import NIOHPACK
import os
import ServiceContextModule
import Tracing

final class CloudBoardProvider: Com_Apple_Cloudboard_Api_V1_CloudBoardAsyncProvider, Sendable {
    public static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "CloudBoardProvider"
    )

    enum Error: Swift.Error {
        case receivedMultipleFinalChunks
        case receivedMultipleDecryptionKeys
        case incomingConnectionClosedEarly
        case receivedRequestAfterRequestStreamTerminated
        case alreadyWaitingForIdle
    }

    static let rpcIDHeaderName = "apple-rpc-uuid"

    // Delegate to handle messages from cb_jobhelper
    actor JobHelperResponseDelegate: CloudBoardJobHelperAPIClientDelegateProtocol {
        let responseContinuation: AsyncStream<CloudBoardJobHelperAPI.WorkloadResponse>.Continuation

        init(responseContinuation: AsyncStream<CloudBoardJobHelperAPI.WorkloadResponse>.Continuation) {
            self.responseContinuation = responseContinuation
        }

        nonisolated func cloudBoardJobHelperAPIClientSurpriseDisconnect() {
            CloudBoardProvider.logger.info("surprise disconnect of job helper client")
            self.responseContinuation.finish()
        }

        func sendWorkloadResponse(_ response: CloudBoardJobHelperAPI.WorkloadResponse) {
            CloudBoardProvider.logger.debug("JobHelperResponseDelegate sendWorkloadResponse")
            self.responseContinuation.yield(response)
            switch response {
            case .responseChunk(let responseChunk):
                if responseChunk.isFinal {
                    CloudBoardProvider.logger.info("JobHelperResponseDelegate sendWorkloadResponse isFinal")
                    self.responseContinuation.finish()
                }
            case .failureReport:
                () // nothing more to do, already yielded
            }
        }
    }

    struct JobHelperResponseDelegateProvider: CloudBoardJobHelperResponseDelegateProvider {
        func makeDelegate(
            responseContinuation: AsyncStream<WorkloadResponse>.Continuation
        ) -> CloudBoardJobHelperAPIClientDelegateProtocol {
            return JobHelperResponseDelegate(responseContinuation: responseContinuation)
        }
    }

    let sessionStore: SessionStore

    let jobHelperClientProvider: CloudBoardJobHelperClientProvider
    let jobHelperResponseDelegateProvider: CloudBoardJobHelperResponseDelegateProvider
    let healthMonitor: ServiceHealthMonitor
    let attestationProvider: AttestationProvider
    let loadState: OSAllocatedUnfairLock<LoadState> = .init(initialState: .init(
        concurrentRequestCount: 0,
        maxConcurrentRequests: 0
    ))
    let loadConfiguration: CloudBoardDConfiguration.LoadConfiguration
    private let hotProperties: HotPropertiesController?
    let metrics: any MetricsSystem
    let tracer: any Tracer

    private let (_concurrentRequestCountStream, concurrentRequestCountContinuation) = AsyncStream
        .makeStream(of: Int.self)
    var concurrentRequestCountStream: AsyncStream<Int> {
        self._concurrentRequestCountStream
    }

    private let drainState = OSAllocatedUnfairLock(initialState: DrainState())
    public var activeRequestsBeforeDrain: Int {
        self.drainState.withLock { $0.activeRequestsBeforeDrain }
    }

    struct DrainState {
        var activeRequests: Int
        var activeRequestsBeforeDrain: Int
        var draining: Bool
        var drainCompleteContinuation: CheckedContinuation<Void, Never>?

        init() {
            self.activeRequests = 0
            self.activeRequestsBeforeDrain = 0
            self.draining = false
            self.drainCompleteContinuation = nil
        }

        mutating func requestStarted() throws {
            if self.draining {
                CloudBoardProvider.logger.warning("Received invokeWorkload gRPC message; but draining")
                throw GRPCTransformableError.drainingRequests
            }
            self.activeRequests += 1
        }

        mutating func requestFinished() {
            self.activeRequests -= 1

            if self.draining, self.activeRequests == 0 {
                CloudBoardProvider.logger
                    .debug("CloudBoardProvider drain has reached 0 active requests")
                if let continuation = self.drainCompleteContinuation {
                    self.drainCompleteContinuation = nil
                    continuation.resume()
                } else {
                    assertionFailure("Missing drain complete continuation")
                }
            }
        }

        mutating func drain() {
            self.draining = true
            self.activeRequestsBeforeDrain = self.activeRequests
        }
    }

    init(
        jobHelperClientProvider: CloudBoardJobHelperClientProvider,
        jobHelperResponseDelegateProvider: CloudBoardJobHelperResponseDelegateProvider,
        healthMonitor: ServiceHealthMonitor,
        metrics: any MetricsSystem,
        tracer: any Tracer,
        attestationProvider: AttestationProvider,
        loadConfiguration: CloudBoardDConfiguration.LoadConfiguration,
        hotProperties: HotPropertiesController?
    ) {
        self.jobHelperClientProvider = jobHelperClientProvider
        self.jobHelperResponseDelegateProvider = jobHelperResponseDelegateProvider
        self.healthMonitor = healthMonitor
        self.attestationProvider = attestationProvider
        self.metrics = metrics
        self.tracer = tracer
        self.loadConfiguration = loadConfiguration
        self.hotProperties = hotProperties
        self.sessionStore = SessionStore(attestationProvider: attestationProvider, metrics: self.metrics)
    }

    func run() async {
        // We start our own watch of the service health here in order to manage
        // max concurrent requests.
        for await update in self.healthMonitor.watch() {
            let maxConcurrentRequests: Int = switch update {
            case .initializing, .unhealthy:
                0
            case .healthy(let state):
                state.maxBatchSize
            }

            self.loadState.withLock {
                $0.maxConcurrentRequests = maxConcurrentRequests
            }
        }
    }

    func invokeWorkload(
        requestStream: GRPCAsyncRequestStream<Com_Apple_Cloudboard_Api_V1_InvokeWorkloadRequest>,
        responseStream: GRPCAsyncResponseStreamWriter<Com_Apple_Cloudboard_Api_V1_InvokeWorkloadResponse>,
        context: GRPCAsyncServerCallContext
    ) async throws {
        let rpcID = self.extractRPCID(from: context.request.headers)

        var serviceContext = ServiceContext.topLevel
        serviceContext.rpcID = rpcID.uuidString
        try await CloudBoardDaemon.$rpcID.withValue(rpcID) {
            try await self.tracer.withSpan(OperationNames.invokeWorkload, context: serviceContext) { span in
                span.attributes.requestSummary.invocationAttributes.invocationRequestHeaders = context.request
                    .headers
                try await withTaskCancellationHandler {
                    try await self.policeDraining {
                        try await self.enforceConcurrentWorkloadLimit {
                            do {
                                try await self._invokeWorkload(requestStream, responseStream)
                            } catch is CancellationError {
                                // If the top-level task is cancelled we were cancelled by grpc-swift due to the
                                // connection/stream having been cancelled. In this case CancellationErrors are expected
                                // and we should classify them accordingly
                                if Task.isCancelled {
                                    throw GRPCTransformableError.connectionCancelled
                                } else {
                                    throw CancellationError()
                                }
                            }
                        }
                    }
                } onCancel: {
                    Self.logger.log("\(Self.logMetadata(), privacy: .public) Connection cancelled")
                    span.attributes.requestSummary.invocationAttributes.connectionCancelled = true
                }
            }
        }
    }

    private func extractRPCID(from headers: HPACKHeaders) -> UUID {
        let parsedRPCID = headers.first(name: Self.rpcIDHeaderName).flatMap {
            UUID(uuidString: $0)
        }
        guard let parsedRPCID else {
            let unavailableRPCID = UUID()
            Self.logger.warning(
                "invokeWorkload() could not parse request id from header \(Self.rpcIDHeaderName, privacy: .public). Using \(unavailableRPCID, privacy: .public) instead."
            )
            return unavailableRPCID
        }
        return parsedRPCID
    }

    fileprivate func enforceConcurrentWorkloadLimit(
        executeWorkload: () async throws -> Void
    ) async throws {
        try self.loadState.withLock { loadState in
            let maxConcurrentRequests = loadState.maxConcurrentRequests
            if loadState.concurrentRequestCount >= maxConcurrentRequests {
                self.metrics.emit(Metrics.CloudBoardProvider.MaxConcurrentRequestCountExceededTotal(action: .increment))

                if self.loadConfiguration.enforceConcurrentRequestLimit {
                    self.metrics.emit(
                        Metrics.CloudBoardProvider.MaxConcurrentRequestCountRejectedTotal(action: .increment)
                    )
                    Self.logger.warning(
                        "\(Self.logMetadata(), privacy: .public) incoming workload rejected because it would exceed the number of max concurrent request count of \(maxConcurrentRequests, privacy: .public)"
                    )
                    if maxConcurrentRequests != 0 {
                        throw GRPCTransformableError.maxConcurrentRequestsExceeded
                    } else {
                        // Oh shoot we're unhealthy!
                        throw GRPCTransformableError.workloadUnhealthy
                    }
                } else {
                    Self.logger.warning(
                        "\(Self.logMetadata(), privacy: .public) incoming workload request exceeds the number of max concurrent request count of \(maxConcurrentRequests, privacy: .public) but accepted as enforcement is disabled"
                    )
                }
            }
            loadState.concurrentRequestCount += 1
            if self.loadConfiguration.overrideCloudAppConcurrentRequests {
                self.healthMonitor.overrideCurrentRequestCount(count: loadState.concurrentRequestCount)
            }
            self.metrics.emit(Metrics.CloudBoardProvider.ConcurrentRequests(value: loadState.concurrentRequestCount))
            self.concurrentRequestCountContinuation.yield(loadState.concurrentRequestCount)
        }
        defer {
            self.loadState.withLock { loadState in
                loadState.concurrentRequestCount -= 1
                if self.loadConfiguration.overrideCloudAppConcurrentRequests {
                    self.healthMonitor.overrideCurrentRequestCount(count: loadState.concurrentRequestCount)
                }
                self.metrics.emit(
                    Metrics.CloudBoardProvider.ConcurrentRequests(value: loadState.concurrentRequestCount)
                )
                concurrentRequestCountContinuation.yield(loadState.concurrentRequestCount)

                if let continuation = loadState.idleContinuation {
                    if loadState.concurrentRequestCount == 0 {
                        loadState.idleContinuation = nil
                        continuation.finish()
                    }
                }
            }
        }

        try await executeWorkload()
    }

    public func pause() async throws {
        let (idleStream, idleContinuation) =
            AsyncStream.makeStream(of: Void.self)

        try self.loadState.withLock { loadState in
            guard loadState.idleContinuation == nil else {
                throw CloudBoardProvider.Error.alreadyWaitingForIdle
            }
            loadState.maxConcurrentRequests = 0
            if loadState.concurrentRequestCount == 0 {
                idleContinuation.finish()
                return
            }
            loadState.idleContinuation = idleContinuation
        }

        for await _ in idleStream {
            return
        }
    }

    fileprivate func policeDraining(
        executeWorkload: () async throws -> Void
    ) async throws {
        // Check for ongoing draining
        try self.drainState.withLock { drainState in
            // Throws if draining is in progress
            try drainState.requestStarted()
        }
        defer {
            self.drainState.withLock { drainState in
                drainState.requestFinished()
            }
        }

        try await executeWorkload()
    }

    fileprivate enum WorkloadTaskResult {
        case jobHelperExited
        case receiveInputCompleted
        case responseOutputCompleted
    }

    fileprivate func _invokeWorkload(
        _ requestStream: GRPCAsyncRequestStream<Com_Apple_Cloudboard_Api_V1_InvokeWorkloadRequest>,
        _ responseStream: GRPCAsyncResponseStreamWriter<Com_Apple_Cloudboard_Api_V1_InvokeWorkloadResponse>
    ) async throws {
        try await withErrorLogging(
            operation: "_invokeWorkload",
            diagnosticKeys: CloudBoardDaemonDiagnosticKeys([.rpcID]),
            sensitiveError: false
        ) {
            CloudBoardProviderCheckpoint(
                logMetadata: Self.logMetadata(),
                message: "invokeWorkload() received workload invocation request"
            ).log(to: Self.logger)

            let (jobHelperResponseStream, jobHelperResponseContinuation) = AsyncStream<WorkloadResponse>.makeStream()

            let delegate = await self.jobHelperResponseDelegateProvider
                .makeDelegate(responseContinuation: jobHelperResponseContinuation)
            let idleTimeout = await self.idleTimeout(
                taskName: "invokeWorkload", taskID: CloudBoardDaemon.rpcID.uuidString
            )
            try await self.jobHelperClientProvider.withClient(delegate: delegate) { jobHelperClient in
                try await withThrowingTaskGroup(of: WorkloadTaskResult.self) { group in
                    group.addTaskWithLogging(
                        operation: "_invokeWorkload.idleTimeout",
                        diagnosticKeys: CloudBoardDaemonDiagnosticKeys([.rpcID]),
                        sensitiveError: false
                    ) {
                        do {
                            try await idleTimeout.run()
                        } catch let error as IdleTimeoutError {
                            CloudBoardProviderCheckpoint(
                                logMetadata: Self.logMetadata(),
                                message: "preparing idle timeout",
                                error: error
                            ).log(to: Self.logger, level: .error)
                            throw GRPCTransformableError(idleTimeoutError: error)
                        }
                    }

                    group.addTaskWithLogging(
                        operation: "_invokeWorkload.requestStream",
                        diagnosticKeys: CloudBoardDaemonDiagnosticKeys([.rpcID]),
                        sensitiveError: false
                    ) {
                        do {
                            try await self.processInvokeWorkloadRequests(
                                requestStream: requestStream,
                                responseStream: responseStream,
                                jobHelperClient: jobHelperClient,
                                idleTimeout: idleTimeout
                            )
                            return .receiveInputCompleted
                        } catch let error as InvokeWorkloadStreamState.Error {
                            throw GRPCTransformableError(error)
                        }
                    }

                    group.addTaskWithLogging(
                        operation: "_invokeWorkload.jobHelperClient",
                        diagnosticKeys: CloudBoardDaemonDiagnosticKeys([.rpcID]),
                        sensitiveError: false
                    ) {
                        try await jobHelperClient.waitForExit()
                        // Now that we no longer have a cb_jobhelper instance, ensure
                        // the output stream is finished.
                        jobHelperResponseContinuation.finish()
                        return .jobHelperExited
                    }

                    group.addTaskWithLogging(
                        operation: "_invokeWorkload.jobHelperResponseStream",
                        diagnosticKeys: CloudBoardDaemonDiagnosticKeys([.rpcID]),
                        sensitiveError: false
                    ) {
                        try await self.tracer.withSpan(OperationNames.invokeWorkloadResponse) { span in
                            for await jobHelperResponse in jobHelperResponseStream {
                                Self.logger.debug(
                                    "\(Self.logMetadata(), privacy: .public) received response from cb_jobhelper"
                                )
                                idleTimeout.registerActivity()

                                switch jobHelperResponse {
                                case .responseChunk(let responseChunk):
                                    span.attributes.requestSummary.responseChunkAttributes
                                        .chunksCount = (
                                            span.attributes.requestSummary.responseChunkAttributes
                                                .chunksCount ?? 0
                                        ) + 1
                                    span.attributes.requestSummary.responseChunkAttributes
                                        .isFinal = (
                                            span.attributes.requestSummary.responseChunkAttributes
                                                .isFinal ?? false
                                        ) || responseChunk.isFinal
                                    // Send our response piece.
                                    try await responseStream.send(.with {
                                        $0.responseChunk = .with {
                                            $0.encryptedPayload = responseChunk.encryptedPayload
                                            $0.isFinal = responseChunk.isFinal
                                        }
                                    })
                                    Self.logger.debug("\(Self.logMetadata(), privacy: .public) sent grpc response")
                                case .failureReport(let failureReason):
                                    CloudBoardProviderCheckpoint(
                                        logMetadata: Self.logMetadata(),
                                        message: "received error response from cb_jobhelper with FailureReason"
                                    ).log(failureReason: failureReason, to: Self.logger)
                                    let pushFailureReportsToROPES: Bool = await self.hotProperties?.currentValue?
                                        .pushFailureReportsToROPES == true
                                    if pushFailureReportsToROPES {
                                        throw GRPCTransformableError(failureReason: failureReason)
                                    }
                                }
                            }
                        }
                        return .responseOutputCompleted
                    }

                    // Once the associated cb_jobhelper exits and we finish sending
                    // any response data, we can cancel any remaining tasks.
                    enum CompletionStatus {
                        case awaitingCompletion
                        case awaitingResponseStreamCompletion
                        case awaitingJobHelperCompletion
                        case completed
                    }
                    var status: CompletionStatus = .awaitingCompletion
                    taskResultLoop: for try await result in group {
                        if group.isCancelled {
                            break
                        }
                        switch result {
                        case .jobHelperExited:
                            if status == .awaitingJobHelperCompletion {
                                status = .completed
                            } else {
                                status = .awaitingResponseStreamCompletion
                            }
                        case .responseOutputCompleted:
                            if status == .awaitingResponseStreamCompletion {
                                status = .completed
                            } else {
                                status = .awaitingJobHelperCompletion
                            }
                        case .receiveInputCompleted:
                            // Nothing to do, job helper is expected to terminate on its own at this point
                            ()
                        }
                        if case .completed = status {
                            CloudBoardProviderCheckpoint(
                                logMetadata: Self.logMetadata(),
                                message: "cb_jobhelper exited + output completed, cancelling remaining work in invokeWorkload"
                            ).log(to: Self.logger)
                            group.cancelAll()
                            break taskResultLoop
                        }
                    }
                }
            }
        }
    }

    private func invokeJobHelperRequest(
        for request: Com_Apple_Cloudboard_Api_V1_InvokeWorkloadRequest,
        stateMachine: inout InvokeWorkloadRequestStateMachine,
        jobHelperClient: CloudBoardJobHelperInstanceProtocol
    ) async throws {
        if case .setup = request.type {
            return
        }
        if let jobHelperRequest = InvokeWorkloadRequest(from: request) {
            try stateMachine.receive(jobHelperRequest)
            try await jobHelperClient.invokeWorkloadRequest(jobHelperRequest)
        } else {
            CloudBoardProviderCheckpoint(
                logMetadata: Self.logMetadata(),
                message: "received invalid InvokeWorkloadRequest message on request stream, ignoring"
            ).log(to: Self.logger, level: .error)
        }
    }

    private func processInvokeWorkloadRequests(
        requestStream: GRPCAsyncRequestStream<Com_Apple_Cloudboard_Api_V1_InvokeWorkloadRequest>,
        responseStream: GRPCAsyncResponseStreamWriter<Com_Apple_Cloudboard_Api_V1_InvokeWorkloadResponse>,
        jobHelperClient: CloudBoardJobHelperInstanceProtocol,
        idleTimeout: IdleTimeout<ContinuousClock>
    ) async throws {
        var cumulativeRequestBytesLimiter = CumulativeRequestBytesLimiter()
        var stateMachine = InvokeWorkloadRequestStateMachine()
        var requestID = ""
        var state = InvokeWorkloadStreamState()
        let maxCumulativeRequestBytes = await self.currentMaxCumulativeRequestBytes()
        for try await request in requestStream {
            idleTimeout.registerActivity()
            try state.receiveMessage(request)

            if let id = extractRequestID(from: request) {
                requestID = id
            }
            try await CloudBoardDaemon.$requestTrackingID.withValue(requestID) {
                try cumulativeRequestBytesLimiter.enforceLimit(
                    maxCumulativeRequestBytes,
                    on: request,
                    metrics: self.metrics
                )

                var nextContext = ServiceContext.$current.get() ?? .topLevel
                nextContext.requestID = requestID
                try await self.tracer.withSpan(OperationNames.invokeWorkloadRequest, context: nextContext) { span in
                    switch request.type {
                    case .setup:
                        CloudBoardProviderCheckpoint(
                            logMetadata: Self.logMetadata(),
                            message: "received workload setup request, awaiting warmup complete before continuing"
                        ).log(to: Self.logger)
                        span.attributes.requestSummary.workloadRequestAttributes.receivedSetup = true
                        let waitForWarmupCompleteTimeMeasurement = ContinuousTimeMeasurement.start()
                        do {
                            try await jobHelperClient.waitForWarmupComplete()
                            self.metrics.emit(
                                Metrics.CloudBoardProvider.WaitForWarmupCompleteTimeHistogram(
                                    duration: waitForWarmupCompleteTimeMeasurement.duration
                                )
                            )
                        } catch {
                            self.metrics.emit(
                                Metrics.CloudBoardProvider.FailedWaitForWarmupCompleteTimeHistogram(
                                    duration: waitForWarmupCompleteTimeMeasurement.duration
                                )
                            )
                            CloudBoardProviderCheckpoint(
                                logMetadata: Self.logMetadata(),
                                message: "waitForWarmupComplete returned error",
                                error: error
                            ).log(to: Self.logger, level: .error)
                            throw error
                        }
                        CloudBoardProviderCheckpoint(
                            logMetadata: Self.logMetadata(),
                            message: "warmup completed, sending acknowledgement"
                        ).log(to: Self.logger)
                        idleTimeout.registerActivity()
                        try await responseStream.send(.with {
                            $0.setupAck = .with { $0.supportTerminate = true }
                        })
                    case .parameters:
                        CloudBoardProviderCheckpoint(
                            logMetadata: Self.logMetadata(),
                            message: "received workload parameters request"
                        ).log(to: Self.logger)

                        span.attributes.requestSummary.workloadRequestAttributes.receivedParameters = true
                        span.attributes.requestSummary.workloadRequestAttributes.bundleID = request.parameters
                            .tenantInfo
                            .bundleID
                        span.attributes.requestSummary.workloadRequestAttributes.featureID = request.parameters
                            .tenantInfo
                            .featureID
                        span.attributes.requestSummary.workloadRequestAttributes.workload = request.parameters.workload
                            .type
                        span.attributes.requestSummary.workloadRequestAttributes.automatedDeviceGroup =
                            request.parameters.tenantInfo.automatedDeviceGroup

                        // Check for replay and reject request if the decryption key has been seen before
                        do {
                            // The wrapped key material of the request is used as a session identifier that must be
                            // unique
                            let keyMaterial = request.parameters.decryptionKey.encryptedPayload
                            let keyID = request.parameters.decryptionKey.keyID
                            try await self.sessionStore.addSession(keyMaterial: keyMaterial, keyID: keyID)
                        } catch {
                            CloudBoardProviderCheckpoint(
                                logMetadata: Self.logMetadata(),
                                message: "Failed to add decryption key to session store",
                                error: error
                            ).log(to: Self.logger, level: .error)
                            throw error
                        }
                    case .requestChunk(let chunk):
                        CloudBoardProviderCheckpoint(
                            logMetadata: Self.logMetadata(),
                            message: "received workload request chunk request"
                        ).log(to: Self.logger)
                        span.attributes.requestSummary.workloadRequestAttributes.chunkSize = chunk.encryptedPayload
                            .count
                        span.attributes.requestSummary.workloadRequestAttributes.isFinal = chunk.isFinal
                    case .terminate(let message):
                        span.attributes.requestSummary.workloadRequestAttributes.ropesTerminationCode = message.code
                            .rawValue
                        span.attributes.requestSummary.workloadRequestAttributes.ropesTerminationReason = message.reason
                        CloudBoardProviderCheckpoint(
                            logMetadata: Self.logMetadata(),
                            message: "Received termination notification from ROPES"
                        ).log(
                            terminationCode: message.code.rawValue,
                            terminationReason: message.reason,
                            to: Self.logger
                        )
                    case .none:
                        break
                    }
                    try await self.invokeJobHelperRequest(
                        for: request,
                        stateMachine: &stateMachine,
                        jobHelperClient: jobHelperClient
                    )
                }
            }
        }

        try state.receiveEOF()
    }

    private func currentMaxCumulativeRequestBytes() async -> Int {
        let defaultValue = self.loadConfiguration.maxCumulativeRequestBytes

        guard let hotProperties = self.hotProperties else {
            return defaultValue
        }

        guard let currentConfigValue = await hotProperties.currentValue else {
            return defaultValue
        }

        return currentConfigValue.maxCumulativeRequestBytes ?? defaultValue
    }

    func watchLoadLevel(
        request _: Com_Apple_Cloudboard_Api_V1_LoadRequest,
        responseStream: GRPCAsyncResponseStreamWriter<Com_Apple_Cloudboard_Api_V1_LoadResponse>,
        context _: GRPCAsyncServerCallContext
    ) async throws {
        try await withErrorLogging(operation: "watchLoadLevel", sensitiveError: false) {
            Self.logger.info("received watch load level request")

            var lastWorkload: Workload?

            for await status in self.healthMonitor.watch() {
                Self.logger.info("new load status: \(status, privacy: .public)")

                switch status {
                case .healthy(let healthy):
                    let newWorkload = Workload(healthy)
                    let sendWorkload = newWorkload != lastWorkload
                    try await responseStream.send(.init(healthy, sendWorkload: sendWorkload))
                    lastWorkload = newWorkload
                case .initializing, .unhealthy:
                    // Unhealthy means load level of 0.
                    try await responseStream.send(.with {
                        $0.currentBatchSize = 0
                        $0.maxBatchSize = 0
                        $0.optimalBatchSize = 0
                    })
                    // Deliberately don't reset the workload: if the same workload comes back, we don't need
                    // to send this value again.
                }
            }
        }
    }

    func fetchAttestation(
        request _: Com_Apple_Cloudboard_Api_V1_FetchAttestationRequest,
        context: InternalGRPC.GRPCAsyncServerCallContext
    ) async throws -> Com_Apple_Cloudboard_Api_V1_FetchAttestationResponse {
        let rpcID = self.extractRPCID(from: context.request.headers)
        Self.logger.log("message=\"received fetch attestation request\"\nrpcID=\(rpcID)")
        return try await withErrorLogging(operation: "fetchAttestation", sensitiveError: false) {
            let requestSummary = FetchAttestationRequestSummary(rpcID: rpcID)
            return try await requestSummary.loggingRequestSummaryModifying(logger: Self.logger) { summary in
                let attestationSet = try await self.attestationProvider.currentAttestationSet()
                summary.populateAttestationSet(attestationSet: attestationSet)
                return .with {
                    $0.attestation = .with {
                        $0.attestationBundle = attestationSet.currentAttestation.attestationBundle
                        $0.keyID = attestationSet.currentAttestation.keyID
                        $0.nextRefreshTime = .init(
                            timeIntervalSince1970: attestationSet.currentAttestation.publicationExpiry
                                .timeIntervalSince1970
                        )
                    }
                    $0.unpublishedAttestation = attestationSet.unpublishedAttestations.map { key in
                        .with {
                            $0.keyID = key.keyID
                        }
                    }
                }
            }
        }
    }

    func drain() async {
        await withCheckedContinuation { drainCompleteContinuation in
            self.drainState.withLock { drainState in
                precondition(!drainState.draining, "CloudBoardProvider received request to drain during ongoing drain")

                drainState.drain()
                if drainState.activeRequests == 0 {
                    drainCompleteContinuation.resume()
                } else {
                    drainState.drainCompleteContinuation = drainCompleteContinuation
                }
            }
        }
    }

    private func idleTimeout(taskName: String, taskID: String) async -> IdleTimeout<ContinuousClock> {
        // If we for any reason can't find a better value, let's use 30s.
        let duration = await Duration.milliseconds(
            self.hotProperties?.currentValue?.idleTimeoutMilliseconds ?? 30 * 1000
        )
        CloudBoardProviderCheckpoint(
            logMetadata: Self.logMetadata(),
            message: "preparing idle timeout"
        ).log(timeoutDuration: duration, to: Self.logger)
        return IdleTimeout(timeout: duration, taskName: taskName, taskID: taskID)
    }
}

struct LoadState {
    var concurrentRequestCount: Int
    var maxConcurrentRequests: Int
    var idleContinuation: AsyncStream<Void>.Continuation?
}

extension InvokeWorkloadRequest {
    init?(from cloudBoardRequest: Com_Apple_Cloudboard_Api_V1_InvokeWorkloadRequest) {
        switch cloudBoardRequest.type {
        case .parameters(let parameters):
            self = .parameters(.init(
                requestID: parameters.requestID,
                oneTimeToken: parameters.oneTimeToken,
                encryptedKey: .init(
                    keyID: parameters.decryptionKey.keyID,
                    key: parameters.decryptionKey.encryptedPayload
                ),
                parametersReceived: .now,
                plaintextMetadata: .init(tenantInfo: parameters.tenantInfo, workload: parameters.workload)
            ))
        case .requestChunk(let requestChunk):
            self = .requestChunk(requestChunk.encryptedPayload, isFinal: requestChunk.isFinal)
        case .none, .setup, .terminate:
            return nil
        }
    }
}

/// State machine to keep track of invoke workload request state. Needed to determine when a request stream ends early,
/// i.e. before we have received a final chunk and a decryption key.
struct InvokeWorkloadRequestStateMachine {
    enum State {
        case awaitingFinalChunkAndDecryptionKey
        case awaitingFinalChunk
        case awaitingDecryptionKey
        case receivedFinalChunkAndDecryptionKey
        case terminated
    }

    private var state: State

    init() {
        self.state = .awaitingFinalChunkAndDecryptionKey
    }

    mutating func receive(_ request: InvokeWorkloadRequest) throws {
        if case .requestChunk(_, let isFinal) = request, isFinal {
            switch self.state {
            case .awaitingFinalChunkAndDecryptionKey:
                self.state = .awaitingDecryptionKey
            case .awaitingFinalChunk:
                self.state = .receivedFinalChunkAndDecryptionKey
            case .awaitingDecryptionKey, .receivedFinalChunkAndDecryptionKey:
                throw CloudBoardProvider.Error.receivedMultipleFinalChunks
            case .terminated:
                CloudBoardProvider.logger.fault("Unexpectedly received request after request stream has terminated")
                throw CloudBoardProvider.Error.receivedRequestAfterRequestStreamTerminated
            }
        } else if case .parameters = request {
            switch self.state {
            case .awaitingFinalChunkAndDecryptionKey:
                self.state = .awaitingFinalChunk
            case .awaitingDecryptionKey:
                self.state = .receivedFinalChunkAndDecryptionKey
            case .awaitingFinalChunk, .receivedFinalChunkAndDecryptionKey:
                throw CloudBoardProvider.Error.receivedMultipleDecryptionKeys
            case .terminated:
                CloudBoardProvider.logger.fault("Unexpectedly received request after request stream has terminated")
                throw CloudBoardProvider.Error.receivedRequestAfterRequestStreamTerminated
            }
        }
    }

    mutating func streamEnded() throws {
        defer {
            self.state = .terminated
        }
        guard case .receivedFinalChunkAndDecryptionKey = self.state else {
            throw CloudBoardProvider.Error.incomingConnectionClosedEarly
        }
    }
}

private struct Workload: Hashable {
    var type: String
    var tags: [String: [String]]

    init(_ status: ServiceHealthMonitor.Status.Healthy) {
        self.type = status.workloadType
        self.tags = status.tags
    }
}

extension Com_Apple_Cloudboard_Api_V1_LoadResponse {
    init(_ status: ServiceHealthMonitor.Status.Healthy, sendWorkload: Bool) {
        self = .with {
            $0.currentBatchSize = UInt32(status.currentBatchSize)
            $0.maxBatchSize = UInt32(status.maxBatchSize)
            $0.optimalBatchSize = UInt32(status.optimalBatchSize)

            if sendWorkload {
                $0.workload = .with {
                    $0.type = status.workloadType
                    $0.param = .init(status.tags)
                }
            }
        }
    }
}

extension [Com_Apple_Cloudboard_Api_V1_Workload.Parameter] {
    init(_ tags: [String: [String]]) {
        self = tags.map { key, values in
            .with {
                $0.key = key
                $0.value = values
            }
        }
    }
}

func extractRequestID(
    from request: Com_Apple_Cloudboard_Api_V1_InvokeWorkloadRequest
) -> String? {
    switch request.type {
    case .parameters(let parameters):
        return parameters.requestID
    default:
        return nil
    }
}

extension CloudBoardProvider {
    internal static func logMetadata() -> CloudBoardDaemonLogMetadata {
        return CloudBoardDaemonLogMetadata(
            rpcID: CloudBoardDaemon.rpcID,
            requestTrackingID: CloudBoardDaemon.requestTrackingID
        )
    }
}

extension Parameters.PlaintextMetadata {
    init(
        tenantInfo: Com_Apple_Cloudboard_Api_V1_TenantInfo,
        workload: Com_Apple_Cloudboard_Api_V1_Workload
    ) {
        self.init(
            bundleID: tenantInfo.bundleID,
            bundleVersion: tenantInfo.bundleVersion,
            featureID: tenantInfo.featureID,
            clientInfo: tenantInfo.clientInfo,
            workloadType: workload.type,
            workloadParameters: Dictionary(parameters: workload.param),
            automatedDeviceGroup: tenantInfo.automatedDeviceGroup
        )
    }
}

extension [String: [String]] {
    init(parameters: [Com_Apple_Cloudboard_Api_V1_Workload.Parameter]) {
        self = [:]

        for parameter in parameters {
            self[parameter.key, default: []].append(contentsOf: parameter.value)
        }
    }
}

/// NOTE: CloudBoardDCore is considered safe for logging the entire error description, so the detailed error
/// is always logged as public in`CloudBoardProviderCheckpoint`.
struct CloudBoardProviderCheckpoint: RequestCheckpoint {
    var requestID: String? {
        self.logMetadata.requestTrackingID
    }

    var operationName: StaticString

    var serviceName: StaticString = "cloudboardd"

    var namespace: StaticString = "cloudboard"

    var error: Error?

    var logMetadata: CloudBoardDaemonLogMetadata

    var message: StaticString

    public init(
        logMetadata: CloudBoardDaemonLogMetadata,
        operationName: StaticString = #function,
        message: StaticString,
        error: Error? = nil
    ) {
        self.logMetadata = logMetadata
        self.operationName = operationName
        self.message = message
        if let error {
            self.error = error
        }
    }

    public func log(to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        rpcId=\(self.logMetadata.rpcID?.uuidString ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        """)
    }

    public func log(failureReason: FailureReason, to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        rpcId=\(self.logMetadata.rpcID?.uuidString ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        failureReason=\(failureReason, privacy: .public)
        """)
    }

    public func log(timeoutDuration: Duration, to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        rpcId=\(self.logMetadata.rpcID?.uuidString ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        timeoutDuration=\(timeoutDuration, privacy: .public)
        """)
    }

    public func log(terminationCode: Int, terminationReason: String, to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        rpcId=\(self.logMetadata.rpcID?.uuidString ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        terminationCode=\(terminationCode, privacy: .public),
        terminationReason=\(terminationReason, privacy: .public)
        """)
    }
}

struct FetchAttestationRequestSummary: RequestSummary {
    let requestID: String? = nil // No request IDs for fetch attestation requests
    let automatedDeviceGroup: String? = nil // No automated device gorup for fetch attestation requests

    var startTimeNanos: Int64?
    var endTimeNanos: Int64?

    let operationName = "FetchAttestation"
    let type = "RequestSummary"
    var serviceName = "cloudboardd"
    var namespace = "cloudboard"

    let rpcID: UUID
    var attestationSet: AttestationSet?
    var error: Error?

    init(rpcID: UUID) {
        self.rpcID = rpcID
    }

    mutating func populateAttestationSet(attestationSet: AttestationSet) {
        self.attestationSet = attestationSet
    }

    func log(to logger: Logger) {
        logger.log("""
        ttl=RequestSummary
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        tracing.start_time_unix_nano=\(self.startTimeNanos ?? 0, privacy: .public)
        tracing.end_time_unix_nano=\(self.endTimeNanos ?? 0, privacy: .public)
        rpcId=\(self.rpcID, privacy: .public)
        request.duration_ms=\(self.durationMicros.map { String($0 / 1000) } ?? "", privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        tracing.status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { "\(String(reportable: $0))" } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        attestationSet=\(self.attestationSet.map { "\($0)" } ?? "", privacy: .public)
        """)
    }
}

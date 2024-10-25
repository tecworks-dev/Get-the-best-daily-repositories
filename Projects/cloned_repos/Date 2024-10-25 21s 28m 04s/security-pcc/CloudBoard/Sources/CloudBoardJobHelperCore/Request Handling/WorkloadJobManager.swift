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

import CloudBoardCommon
import CloudBoardJobAPI
import CloudBoardJobHelperAPI
import CloudBoardLogging
import CloudBoardMetrics
import Foundation
import InternalSwiftProtobuf
import os

/// Job manager responsible for the communication with privatecloudcomputed on the client and unwrapping/wrapping the
/// application request and response payloads to and from the workload respectively.
final class WorkloadJobManager {
    private typealias PrivateCloudComputeRequest = Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeRequest
    private typealias PrivateCloudComputeResponse = Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse

    private static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "WorkloadJobManager"
    )

    private let requestStream: AsyncStream<PipelinePayload<Data>>
    private let responseContinuation: AsyncStream<FinalizableChunk<Data>>.Continuation
    private let cloudAppRequestContinuation: AsyncStream<PipelinePayload<Data>>.Continuation
    private let cloudAppResponseStream: AsyncThrowingStream<CloudAppResponse, Error>
    private let cloudAppResponseContinuation: AsyncThrowingStream<CloudAppResponse, Error>.Continuation

    private let jobUUID: UUID
    private let uuid: UUID
    private var stateMachine: WorkloadJobStateMachine
    private var buffer: LengthPrefixBuffer

    /// The metrics system to use.
    private let metrics: MetricsSystem

    private var requestPlaintextMetadata: ParametersData.PlaintextMetadata?
    private var requestParametersReceivedInstant: RequestSummaryClock.Timestamp?
    private var requestMessageCount: OSAllocatedUnfairLock<Int> = .init(initialState: 0)
    private var responseMessageCount: OSAllocatedUnfairLock<Int> = .init(initialState: 0)

    init(
        tgtValidator: TokenGrantingTokenValidator,
        enforceTGTValidation: Bool,
        requestStream: AsyncStream<PipelinePayload<Data>>,
        maxRequestMessageSize: Int,
        responseContinuation: AsyncStream<FinalizableChunk<Data>>.Continuation,
        cloudAppRequestContinuation: AsyncStream<PipelinePayload<Data>>.Continuation,
        cloudAppResponseStream: AsyncThrowingStream<CloudAppResponse, Error>,
        cloudAppResponseContinuation: AsyncThrowingStream<CloudAppResponse, Error>.Continuation,
        metrics: MetricsSystem,
        jobUUID: UUID
    ) {
        self.requestStream = requestStream
        self.responseContinuation = responseContinuation
        self.cloudAppRequestContinuation = cloudAppRequestContinuation
        self.cloudAppResponseStream = cloudAppResponseStream
        self.cloudAppResponseContinuation = cloudAppResponseContinuation
        self.metrics = metrics

        self.jobUUID = jobUUID
        self.uuid = UUID()
        self.stateMachine = WorkloadJobStateMachine(
            tgtValidator: tgtValidator,
            enforceTGTValidation: enforceTGTValidation,
            metrics: metrics,
            jobUUID: self.jobUUID
        )
        self.buffer = LengthPrefixBuffer(maxMessageSize: maxRequestMessageSize)
    }

    internal func run() async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                defer {
                    self.cloudAppRequestContinuation.finish()
                }
                do {
                    for await message in self.requestStream {
                        self.requestMessageCount.withLock {
                            $0 += 1
                        }
                        try self.receivePipelineMessage(message)
                    }
                    WorkloadJobManagerCheckpoint(
                        logMetadata: self.logMetadata(),
                        requestMessageCount: self.requestMessageCount.withLock { $0 },
                        responseMessageCount: self.responseMessageCount.withLock { $0 },
                        message: "Request stream finished"
                    ).log(to: Self.logger, level: .default)
                    try self.stateMachine.terminate()
                } catch {
                    WorkloadJobManagerCheckpoint(
                        logMetadata: self.logMetadata(),
                        requestMessageCount: self.requestMessageCount.withLock { $0 },
                        responseMessageCount: self.responseMessageCount.withLock { $0 },
                        message: "Error handling request stream",
                        error: error
                    ).log(to: Self.logger, level: .error)
                    // Inject error into response stream to be handled there
                    self.cloudAppResponseContinuation.finish(throwing: error)
                }
            }

            group.addTask {
                var requestSummary = CloudBoardJobHelperRequestSummary(jobUUID: self.jobUUID)
                defer {
                    self.requestPlaintextMetadata.map { requestSummary.populateRequestMetadata($0) }
                    requestSummary.startTimeNanos = self.requestParametersReceivedInstant
                    requestSummary.endTimeNanos = RequestSummaryClock.now
                    requestSummary.requestMessageCount = self.requestMessageCount.withLock { $0 }
                    requestSummary.responseMessageCount = self.responseMessageCount.withLock { $0 }
                    if requestSummary.requestMessageCount > 0 {
                        if requestSummary.responseMessageCount > 0 {
                            self.metrics.emit(Metrics.WorkloadManager.TotalResponsesSentCounter(action: .increment))
                            if let error = requestSummary.error {
                                self.metrics.emit(Metrics.WorkloadManager.FailureResponsesSentCounter.Factory().make(error))
                                // requestDuration should always be set here, but prefer not to crash for telemetry
                                if let durationMicros = requestSummary.durationMicros {
                                    self.metrics.emit(
                                        Metrics.WorkloadManager.WorkloadDurationFromFirstRequestMessage(
                                            duration: .microseconds(durationMicros),
                                            error: error
                                        )
                                    )
                                } else {
                                    WorkloadJobManagerCheckpoint(
                                        logMetadata: self.logMetadata(),
                                        requestMessageCount: self.requestMessageCount.withLock { $0 },
                                        responseMessageCount: self.responseMessageCount.withLock { $0 },
                                        message: "Failed to report workload duration metric - duration was nil",
                                        error: error
                                    ).log(to: Self.logger, level: .error)
                                }
                            } else {
                                self.metrics.emit(Metrics.WorkloadManager.SuccessResponsesSentCounter(action: .increment))
                                // requestDuration should always be set here, but prefer not to crash for telemetry
                                if let durationMicros = requestSummary.durationMicros {
                                    self.metrics.emit(
                                        Metrics.WorkloadManager.WorkloadDurationFromFirstRequestMessage(
                                            duration: .microseconds(durationMicros),
                                            error: nil
                                        )
                                    )
                                } else {
                                    WorkloadJobManagerCheckpoint(
                                        logMetadata: self.logMetadata(),
                                        requestMessageCount: self.requestMessageCount.withLock { $0 },
                                        responseMessageCount: self.responseMessageCount.withLock { $0 },
                                        message: "Failed to report workload duration metric - duration was nil"
                                    ).log(to: Self.logger, level: .error)
                                }
                            }
                        } else { // no response sent, but an error recorded
                            if let error = requestSummary.error {
                                self.metrics.emit(Metrics.WorkloadManager.OverallErrorCounter.Factory().make(error))
                            } else {
                                let error = WorkloadJobManagerNoResponseSentError()
                                requestSummary.populate(error: error)
                                self.metrics.emit(Metrics.WorkloadManager.OverallErrorCounter.Factory().make(error))
                            }
                        }
                    } else {
                        self.metrics.emit(Metrics.WorkloadManager.UnusedTerminationCounter(action: .increment))
                    }
                    requestSummary.log(to: Self.logger)
                }
                defer {
                    self.responseContinuation.finish()
                }
                do {
                    WorkloadJobManagerCheckpoint(
                        logMetadata: self.logMetadata(),
                        requestMessageCount: self.requestMessageCount.withLock { $0 },
                        responseMessageCount: self.responseMessageCount.withLock { $0 },
                        message: "Sending back response UUID"
                    ).log(to: Self.logger, level: .default)
                    try self.responseContinuation.yield(.init(chunk: self.uuidChunk()))

                    // Wait for workload responses, encode them, and forward them
                    var requestDiagnostics = false
                    for try await response in self.cloudAppResponseStream {
                        let responseMessagesReceived = self.responseMessageCount.withLock {
                            $0 += 1
                            return $0
                        }
                        switch response {
                        case .chunk(let data):
                            // downsample the checkpoints we log for response chunks
                            if responseMessagesReceived <= 2 || responseMessagesReceived % 100 == 0 {
                                WorkloadJobManagerCheckpoint(
                                    logMetadata: self.logMetadata(),
                                    requestMessageCount: self.requestMessageCount.withLock { $0 },
                                    responseMessageCount: self.responseMessageCount.withLock { $0 },
                                    message: "Received cloud app response"
                                ).log(to: Self.logger, level: .default)
                            }
                            var serializedResponse = try PrivateCloudComputeResponse.with {
                                $0.type = .responsePayload(data)
                            }.serializedData()
                            serializedResponse.prependLength()
                            self.responseContinuation.yield(.init(chunk: serializedResponse))
                        case .appTermination(let terminationMetadata):
                            WorkloadJobManagerCheckpoint(
                                logMetadata: self.logMetadata(),
                                requestMessageCount: self.requestMessageCount.withLock { $0 },
                                responseMessageCount: self.responseMessageCount.withLock { $0 },
                                message: "Received cloud app termination metadata"
                            ).logAppTermination(
                                terminationMetadata: terminationMetadata,
                                to: Self.logger,
                                level: .default
                            )

                            if let statusCode = terminationMetadata.statusCode, statusCode != 0 {
                                requestDiagnostics = true
                            }
                        }
                    }

                    WorkloadJobManagerCheckpoint(
                        logMetadata: self.logMetadata(),
                        requestMessageCount: self.requestMessageCount.withLock { $0 },
                        responseMessageCount: self.responseMessageCount.withLock { $0 },
                        message: "Cloud app response stream finished"
                    ).log(to: Self.logger, level: .default)

                    do {
                        try self.responseContinuation.yield(.init(
                            chunk: self.responseSummaryChunk(requestDiagnostics: requestDiagnostics),
                            isFinal: true
                        ))
                    } catch {
                        WorkloadJobManagerCheckpoint(
                            logMetadata: self.logMetadata(),
                            requestMessageCount: self.requestMessageCount.withLock { $0 },
                            responseMessageCount: self.responseMessageCount.withLock { $0 },
                            message: "Failed to send response summary",
                            error: error
                        ).log(to: Self.logger, level: .error)
                    }
                    self.metrics.emit(Metrics.WorkloadManager.SuccessResponsesSentCounter(action: .increment))
                } catch {
                    requestSummary.populate(error: error)
                    WorkloadJobManagerCheckpoint(
                        logMetadata: self.logMetadata(),
                        requestMessageCount: self.requestMessageCount.withLock { $0 },
                        responseMessageCount: self.responseMessageCount.withLock { $0 },
                        message: "Error while processing request",
                        error: error
                    ).log(to: Self.logger, level: .error)
                    do {
                        try self.responseContinuation.yield(
                            .init(chunk: self.errorResponseSummaryChunk(for: error), isFinal: true)
                        )
                    } catch {
                        WorkloadJobManagerCheckpoint(
                            logMetadata: self.logMetadata(),
                            requestMessageCount: self.requestMessageCount.withLock { $0 },
                            responseMessageCount: self.responseMessageCount.withLock { $0 },
                            message: "Unexpectedly failed to serialize error response. Not sending error response summary",
                            error: error
                        ).log(to: Self.logger, level: .error)
                    }
                }
            }

            await group.waitForAll()
        }
    }

    private func receivePipelineMessage(_ pipelineMessage: PipelinePayload<Data>) throws {
        self.metrics.emit(Metrics.WorkloadManager.TotalRequestsReceivedCounter(action: .increment))
        // logged via defer block to allow for metadata like requestId to be populated from received messages
        defer {
            WorkloadJobManagerCheckpoint(
                logMetadata: self.logMetadata(),
                requestMessageCount: self.requestMessageCount.withLock { $0 },
                responseMessageCount: self.responseMessageCount.withLock { $0 },
                message: "received pipeline message"
            ).logReceiveRequestPipelineMessage(pipelineMessage: pipelineMessage, to: Self.logger, level: .default)
        }
        switch pipelineMessage {
        case .warmup(let warmupData):
            self.cloudAppRequestContinuation.yield(.warmup(warmupData))
        case .oneTimeToken(let token):
            try self.stateMachine.receiveOneTimeToken(token)
        case .parameters(let parametersData):
            self.requestParametersReceivedInstant = RequestSummaryClock.now
            self.requestPlaintextMetadata = parametersData.plaintextMetadata
            self.stateMachine.receiveRequestID(requestID: parametersData.plaintextMetadata.requestID)
            self.cloudAppRequestContinuation.yield(.parameters(parametersData))
        case .chunk(let finalizableChunk):
            try self.receiveEncodedRequest(finalizableChunk)
        case .endOfInput:
            // Unexpected, ``endOfInput`` is only used between ``WorkloadJobManager`` and the cloud app in response to
            // an encoded request with PrivateCloudCompute.FinalMessage.
            ()
        case .teardown:
            self.cloudAppRequestContinuation.yield(.teardown)
        }
    }

    private func receiveEncodedRequest(_ encodedRequestChunk: FinalizableChunk<Data>) throws {
        for chunk in try self.buffer.append(encodedRequestChunk) {
            switch try PrivateCloudComputeRequest(serializedData: chunk.chunk).type {
            case .applicationPayload(let payload):
                if let cloudAppRequest = stateMachine.receiveChunk(FinalizableChunk(
                    chunk: payload,
                    isFinal: chunk.isFinal
                )) {
                    self.cloudAppRequestContinuation.yield(.chunk(cloudAppRequest))
                }
            case .authToken(let token):
                WorkloadJobManagerCheckpoint(
                    logMetadata: self.logMetadata(),
                    requestMessageCount: self.requestMessageCount.withLock { $0 },
                    responseMessageCount: self.responseMessageCount.withLock { $0 },
                    message: "Received auth token"
                ).log(to: Self.logger, level: .default)

                for cloudAppRequest in try self.stateMachine.receiveAuthToken(token) {
                    switch cloudAppRequest {
                    case .chunk(let chunk):
                        self.cloudAppRequestContinuation.yield(.chunk(chunk))
                    case .finalMessage:
                        self.cloudAppRequestContinuation.yield(.endOfInput)
                    }
                }
            case .finalMessage:
                // This is a message without payload allowing privatecloudcomputed to explicitly indicate the end of the
                // request stream
                WorkloadJobManagerCheckpoint(
                    logMetadata: self.logMetadata(),
                    requestMessageCount: self.requestMessageCount.withLock { $0 },
                    responseMessageCount: self.responseMessageCount.withLock { $0 },
                    message: "Received final message"
                ).log(to: Self.logger, level: .default)
                self.cloudAppRequestContinuation.yield(.endOfInput)
            case .none:
                WorkloadJobManagerCheckpoint(
                    logMetadata: self.logMetadata(),
                    requestMessageCount: self.requestMessageCount.withLock { $0 },
                    responseMessageCount: self.responseMessageCount.withLock { $0 },
                    message: "Received encoded request of unknown type, ignoring"
                ).log(to: Self.logger, level: .debug)
            }
        }
    }

    func uuidChunk() throws -> Data {
        let uuidMessage = PrivateCloudComputeResponse.with {
            $0.type = .responseUuid(withUnsafeBytes(of: self.uuid.uuid) { Data($0) })
        }
        var serialized = try uuidMessage.serializedData()
        serialized.prependLength()
        return serialized
    }

    func responseSummaryChunk(requestDiagnostics: Bool) throws -> Data {
        let summary: PrivateCloudComputeResponse = .with {
            $0.responseSummary = .with {
                $0.responseStatus = .ok
                if requestDiagnostics {
                    $0.postResponseActions = .with {
                        $0.requestDiagnostics = requestDiagnostics
                    }
                }
            }
        }
        var serializedResult = try summary.serializedData()
        serializedResult.prependLength()
        return serializedResult
    }

    func errorResponseSummaryChunk(for error: Error) throws -> Data {
        let summary = PrivateCloudComputeResponse.with {
            $0.responseSummary = .with {
                $0.responseStatus = error.responseStatus
            }
        }
        var serializedResult = try summary.serializedData()
        serializedResult.prependLength()
        return serializedResult
    }
}

enum LengthPrefixedBufferError: Error {
    case exceedingMaxMessageSize(maxSize: Int, announcedSize: Int)
    case receivedAdditionalChunkAfterFinalChunk
    case finalChunkContainsIncompleteMessage
}

struct LengthPrefixBuffer {
    enum State {
        case waitingForLengthPrefix
        case waitingForData(Int)
        case finalChunkSeen
    }

    var buffer: Data
    var state: State

    /// maximum message size in bytes that we are willing to buffer in-memory
    let maxMessageSize: Int

    init(maxMessageSize: Int) {
        self.buffer = Data()
        self.state = .waitingForLengthPrefix
        self.maxMessageSize = maxMessageSize
    }

    mutating func append(_ chunkFragment: FinalizableChunk<Data>) throws -> [FinalizableChunk<Data>] {
        self.buffer.append(chunkFragment.chunk)

        var result: [FinalizableChunk<Data>] = []
        while !self.buffer.isEmpty {
            switch self.state {
            case .waitingForLengthPrefix:
                guard let length = buffer.getLength() else {
                    if chunkFragment.isFinal {
                        throw LengthPrefixedBufferError.finalChunkContainsIncompleteMessage
                    }
                    return result
                }
                guard length <= self.maxMessageSize else {
                    throw LengthPrefixedBufferError.exceedingMaxMessageSize(
                        maxSize: self.maxMessageSize,
                        announcedSize: length
                    )
                }
                self.buffer = self.buffer.dropFirst(4)
                self.state = .waitingForData(length)
            case .waitingForData(let length):
                guard self.buffer.count >= length else {
                    if chunkFragment.isFinal {
                        throw LengthPrefixedBufferError.finalChunkContainsIncompleteMessage
                    }
                    return result
                }

                if chunkFragment.isFinal, self.buffer.count == length {
                    // this is the last chunk and we are at the last message in that chunk
                    result.append(.init(chunk: self.buffer, isFinal: true))
                    self.buffer = Data()
                    self.state = .finalChunkSeen
                } else {
                    // This might be the last chunk fragment but it contains multiple messages.
                    // This is then not yet the last message and therefore we need to set isFinal to false and
                    // parse the the next message in the following iteration
                    // It might also not be the last chunk in which case we set isFinal false too
                    result.append(.init(chunk: self.buffer.prefix(length), isFinal: false))
                    self.buffer = self.buffer.dropFirst(length)
                    self.state = .waitingForLengthPrefix
                }

            case .finalChunkSeen:
                throw LengthPrefixedBufferError.receivedAdditionalChunkAfterFinalChunk
            }
        }

        return result
    }
}

struct WorkloadJobStateMachine {
    private static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "WorkloadJobStateMachine"
    )

    enum BufferedMessage<T> {
        case chunk(FinalizableChunk<T>)
        case finalMessage
    }

    internal enum State {
        case awaitingOneTimeToken([BufferedMessage<Data>])
        case awaitingTokenGrantingToken([BufferedMessage<Data>], oneTimeToken: Data)
        case validatedTokenGrantingToken
        case terminated
    }

    private let tgtValidator: TokenGrantingTokenValidator
    private let enforceTGTValidation: Bool
    private let metrics: MetricsSystem
    private var state: State

    private var requestID: String = ""
    private let jobUUID: UUID

    init(
        tgtValidator: TokenGrantingTokenValidator,
        enforceTGTValidation: Bool,
        metrics: MetricsSystem,
        jobUUID: UUID
    ) {
        self.tgtValidator = tgtValidator
        self.enforceTGTValidation = enforceTGTValidation
        self.metrics = metrics
        self.state = .awaitingOneTimeToken([])
        self.jobUUID = jobUUID
    }

    mutating func receiveChunk(_ chunk: FinalizableChunk<Data>) -> FinalizableChunk<Data>? {
        WorkloadJobStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            state: self.state,
            operation: "receiveChunk"
        ).loggingStateChange(to: Self.logger, level: .debug) {
            switch self.state {
            case .awaitingOneTimeToken(var bufferedMessages):
                bufferedMessages.append(.chunk(chunk))
                self.state = .awaitingOneTimeToken(bufferedMessages)
                return (nil, self.state)
            case .awaitingTokenGrantingToken(var bufferedMessages, let ott):
                bufferedMessages.append(.chunk(chunk))
                self.state = .awaitingTokenGrantingToken(bufferedMessages, oneTimeToken: ott)
                return (nil, self.state)
            case .validatedTokenGrantingToken:
                return (chunk, self.state)
            case .terminated:
                preconditionFailure("Received chunk while already terminated")
            }
        }
    }

    mutating func receiveOneTimeToken(_ oneTimeToken: Data) throws {
        try WorkloadJobStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            state: self.state,
            operation: "receiveOneTimeToken"
        ).loggingStateChange(to: Self.logger, level: .debug) {
            switch self.state {
            case .awaitingOneTimeToken(let chunks):
                self.state = .awaitingTokenGrantingToken(chunks, oneTimeToken: oneTimeToken)
                return ((), self.state)
            case .awaitingTokenGrantingToken, .validatedTokenGrantingToken:
                throw TokenGrantingTokenError.receivedOneTimeTokenTwice
            case .terminated:
                preconditionFailure("Received one-time token while already terminated")
            }
        }
    }

    mutating func receiveTokenGrantingToken(_ tokenGrantingToken: Data) throws -> [BufferedMessage<Data>] {
        try WorkloadJobStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            state: self.state,
            operation: "receiveTokenGrantingToken"
        ).loggingStateChange(to: Self.logger, level: .debug) {
            switch self.state {
            case .awaitingOneTimeToken:
                // Unexpected, we should always receive the one-time token before the token granting token
                throw TokenGrantingTokenError.missingOneTimeToken
            case .awaitingTokenGrantingToken(let bufferedMessages, let ott):
                // Use a 0-byte salt. This will fail verification but we have no OTT salt when clients send the
                // deprecated
                // tokenGrantingToken message instead of the newer authToken message.
                try self.validateToken(tgt: tokenGrantingToken, ott: ott, ottSalt: Data())
                self.state = .validatedTokenGrantingToken
                return (bufferedMessages, self.state)
            case .validatedTokenGrantingToken:
                throw TokenGrantingTokenError.receivedTokenGrantingTokenTwice
            case .terminated:
                preconditionFailure("Received token granting token while already terminated")
            }
        }
    }

    mutating func receiveAuthToken(
        _ authToken: Com_Apple_Privatecloudcompute_Api_V1_AuthToken
    ) throws -> [BufferedMessage<Data>] {
        try WorkloadJobStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            state: self.state,
            operation: "receiveAuthToken"
        ).loggingStateChange(to: Self.logger, level: .debug) {
            switch self.state {
            case .awaitingOneTimeToken:
                // Unexpected, we should always receive the one-time token before the token granting token
                throw TokenGrantingTokenError.missingOneTimeToken
            case .awaitingTokenGrantingToken(let bufferedMessages, let ott):
                try self.validateToken(tgt: authToken.tokenGrantingToken, ott: ott, ottSalt: authToken.ottSalt)
                self.state = .validatedTokenGrantingToken
                return (bufferedMessages, self.state)
            case .validatedTokenGrantingToken:
                throw TokenGrantingTokenError.receivedTokenGrantingTokenTwice
            case .terminated:
                preconditionFailure("Received auth token while already terminated")
            }
        }
    }

    mutating func terminate() throws {
        _ = try WorkloadJobStateMachineCheckpoint(
            logMetadata: self.logMetadata(),
            state: self.state,
            operation: "terminate"
        ).loggingStateChange(to: Self.logger, level: .debug) {
            switch self.state {
            case .awaitingOneTimeToken:
                // Reached the end of the request stream without receiving the one-time token from ROPES
                throw TokenGrantingTokenError.missingOneTimeToken
            case .awaitingTokenGrantingToken:
                // Reached the end of the request stream without receiving a TGT
                throw TokenGrantingTokenError.missingTokenGrantingToken
            case .validatedTokenGrantingToken:
                () // Nothing to do
            case .terminated:
                preconditionFailure("Attempted to terminate more than once")
            }
            self.state = .terminated
            return ((), self.state)
        }
    }

    private func validateToken(tgt: Data, ott: Data, ottSalt: Data) throws {
        do {
            self.metrics.emit(Metrics.WorkloadManager.TGTValidationCounter(action: .increment))
            try self.tgtValidator.validateTokenGrantingToken(tgt, ott: ott, ottSalt: ottSalt)
        } catch {
            self.metrics.emit(Metrics.WorkloadManager.TGTValidationErrorCounter.Factory().make(error))
            if self.enforceTGTValidation {
                throw error
            } else {
                var checkpoint = WorkloadJobStateMachineCheckpoint(
                    logMetadata: self.logMetadata(),
                    state: self.state,
                    operation: "validateTokenError"
                )
                checkpoint.error = error
                checkpoint.log(to: Self.logger, level: .error)
            }
        }
    }

    mutating func receiveRequestID(requestID: String) {
        self.requestID = requestID
    }

    private func logMetadata() -> CloudBoardJobHelperLogMetadata {
        return CloudBoardJobHelperLogMetadata(
            jobID: self.jobUUID,
            requestTrackingID: self.requestID
        )
    }
}

extension Data {
    fileprivate func getLength() -> Int? {
        guard self.count >= 4 else { return nil }
        // swiftformat:disable all
        return Int(
            UInt32(self[startIndex])     << 24 |
            UInt32(self[startIndex + 1]) << 16 |
            UInt32(self[startIndex + 2]) <<  8 |
            UInt32(self[startIndex + 3])
        )
    }

    fileprivate mutating func prependLength() {
        let length = UInt32(self.count)
        let copy = self

        self = Data()
        // swiftformat:disable all
        self.append(contentsOf: [
            UInt8(truncatingIfNeeded: length >> 24),
            UInt8(truncatingIfNeeded: length >> 16),
            UInt8(truncatingIfNeeded: length >>  8),
            UInt8(truncatingIfNeeded: length)
        ])
        // swiftformat:enable all
        self.append(contentsOf: copy)
    }
}

enum TokenGrantingTokenError: Error {
    case missingTokenGrantingToken
    case receivedTokenGrantingTokenTwice
    case missingOneTimeToken
    case receivedOneTimeTokenTwice
}

protocol ResponseStatusConvertible: Error {
    var responseStatus: Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse.ResponseStatus { get }
}

extension Error {
    var responseStatus: Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse.ResponseStatus {
        switch self {
        case let convertibleError as ResponseStatusConvertible:
            return convertibleError.responseStatus
        default:
            return .internalError
        }
    }
}

extension TokenGrantingTokenError: ResponseStatusConvertible {
    var responseStatus: Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse.ResponseStatus {
        switch self {
        case .missingTokenGrantingToken, .missingOneTimeToken:
            return .unauthenticated
        case .receivedTokenGrantingTokenTwice, .receivedOneTimeTokenTwice:
            return .invalidRequest
        }
    }
}

extension TokenGrantingTokenValidationError: ResponseStatusConvertible {
    var responseStatus: Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse.ResponseStatus {
        return .unauthenticated
    }
}

struct WorkloadJobManagerNoResponseSentError: Error {}

extension WorkloadJobManager {
    private func logMetadata() -> CloudBoardJobHelperLogMetadata {
        return CloudBoardJobHelperLogMetadata(
            jobID: self.jobUUID,
            requestTrackingID: self.requestPlaintextMetadata?.requestID
        )
    }
}

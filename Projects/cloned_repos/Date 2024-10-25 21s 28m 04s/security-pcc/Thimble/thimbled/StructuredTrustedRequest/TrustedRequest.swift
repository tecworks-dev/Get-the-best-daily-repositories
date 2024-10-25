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
//  TrustedRequest.swift
//  PrivateCloudCompute
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import AtomicsInternal
@_spi(HTTP) @_spi(OHTTP) import Network
import OSAnalytics
import PrivateCloudCompute
import os

import struct Foundation.Data
import struct Foundation.Date
import struct Foundation.UUID

package protocol IncomingUserDataReaderProtocol: Sendable {
    func forwardData(_ data: Data) async throws

    func ready()
    func waiting()
    func finish(error: (any Error)?)
}

package protocol OutgoingUserDataWriterProtocol: Sendable {
    func withNextOutgoingElement<Result>(_ closure: (OutgoingUserData) async throws -> Result) async throws -> Result

    func cancelAllWrites(error: any Error)
}

enum TrustedRequestConstants {
    static let maxDataToSendBeforeNodeSelected = 65536
}

final class TrustedRequest<
    OutgoingUserDataWriter: OutgoingUserDataWriterProtocol,
    IncomingUserDataReader: IncomingUserDataReaderProtocol,
    ConnectionFactory: NWAsyncConnectionFactoryProtocol,
    AttestationStore: TC2AttestationStoreProtocol,
    AttestationVerifier: TC2AttestationVerifier,
    RateLimiter: RateLimiterProtocol,
    TokenProvider: TC2TokenProvider,
    Clock: _Concurrency.Clock
>: Sendable where Clock.Duration == Duration {
    let clientRequestID: UUID
    let serverRequestID: UUID
    let configuration: TrustedRequestConfiguration
    let parameters: TC2RequestParameters

    let outgoingUserDataWriter: OutgoingUserDataWriter
    let incomingUserDataReader: IncomingUserDataReader
    let connectionFactory: ConnectionFactory
    let attestationStore: AttestationStore?
    let attestationVerifier: AttestationVerifier
    let rateLimiter: RateLimiter
    let tokenProvider: TokenProvider
    let clock: Clock
    let jsonEncoder = tc2JSONEncoder()

    let eventStreamContinuation: AsyncStream<ThimbledEvent>.Continuation
    let requestMetrics: RequestMetrics<Clock, AttestationStore>

    // The request body's OHTTP context
    private let requestOHTTPContext: UInt64 = 1

    private let logger = tc2Logger(forCategory: .TrustedRequest)
    private let lp: LogPrefix

    init(
        clientRequestID: UUID,
        serverRequestID: UUID,
        configuration: TrustedRequestConfiguration,
        parameters: TC2RequestParameters,
        outgoingUserDataWriter: OutgoingUserDataWriter,
        incomingUserDataReader: IncomingUserDataReader,
        connectionFactory: ConnectionFactory,
        attestationStore: AttestationStore?,
        attestationVerifier: AttestationVerifier,
        rateLimiter: RateLimiter,
        tokenProvider: TokenProvider,
        clock: Clock,
        eventStreamContinuation: AsyncStream<ThimbledEvent>.Continuation
    ) {
        self.clientRequestID = clientRequestID
        self.serverRequestID = serverRequestID
        self.lp = LogPrefix(requestID: serverRequestID)

        self.configuration = configuration
        self.parameters = parameters

        self.outgoingUserDataWriter = outgoingUserDataWriter
        self.incomingUserDataReader = incomingUserDataReader
        self.connectionFactory = connectionFactory
        self.attestationStore = attestationStore
        self.attestationVerifier = attestationVerifier
        self.rateLimiter = rateLimiter
        self.tokenProvider = tokenProvider
        self.clock = clock
        self.eventStreamContinuation = eventStreamContinuation
        self.requestMetrics = RequestMetrics(
            clientRequestID: self.clientRequestID,
            serverRequestID: self.serverRequestID,
            bundleID: configuration.bundleID,
            originatingBundleID: configuration.originatingBundleID,
            featureID: configuration.featureID,
            sessionID: configuration.sessionID,
            environment: configuration.environment,
            qos: configuration.serverQoS,
            parameters: parameters,
            logger: self.logger,
            eventStreamContinuation: eventStreamContinuation,
            clock: clock,
            store: self.attestationStore
        )
    }

    func run() async throws {
        self.logger.debug("Running TrustedRequest")
        self.logger.debug("\(self.lp): Configuration: \(self.configuration)")

        try await PowerAssertion.withPowerAssertion(name: "TC2TrustedRequest") {
            do {
                try await self.runRequest()
                self.incomingUserDataReader.finish(error: nil)
                await self.requestMetrics.requestFinished(error: nil)
            } catch {
                // wrapping the internal error as TrustedCloudComputeError for reporting and for throwing to our clients
                let trustedCloudComputeError = TrustedCloudComputeError.wrapAny(error: error)
                self.logger.error("\(self.lp): sendRopesRequest trustedCloudComputeError: \(trustedCloudComputeError) from raw error: \(error)")
                self.incomingUserDataReader.finish(error: trustedCloudComputeError)
                self.outgoingUserDataWriter.cancelAllWrites(error: trustedCloudComputeError)
                await self.requestMetrics.requestFinished(error: trustedCloudComputeError)
                throw trustedCloudComputeError
            }
        }
    }

    // MARK: - Private Methods -

    private enum RunSubTask {
        case ropesRequestDidFinish(Result<Void, any Error>)
        case dataSubStreamDidFinished(Result<Void, any Error>)
        case nodeSubStreamsDidFinished(Result<Void, any Error>)
        case connectionMetricsReportingFinished
    }

    private func runRequest() async throws {
        let sessionCount = try await self.checkRateLimiting()

        try await self.connectionFactory.connect(
            parameters: .makeTLSAndHTTPParameters(
                ignoreCertificateErrors: self.configuration.ignoreCertificateErrors,
                forceOHTTP: self.configuration.forceOHTTP,
                bundleIdentifier: self.configuration.bundleID
            ),
            endpoint: .url(self.configuration.endpointURL),
            activity: nil,  // rdar://127903135 (NWActivity for `computeRequest` and `attestationFetch` were lost in structured request)
            on: .main,
            requestID: self.serverRequestID
        ) { inbound, outbound, ohttpStreams in

            // start network activities first
            self.requestMetrics.attachNetworkActivities(ohttpStreams)

            // We load the one time token (ott) and the cached attestations inside the connection
            // block, since we want to parallelize connection startup, loading the ott and loading
            // the cached attestations to minimize wait time for the user.
            // Remember that the inner block is called before the connection has been established.
            // We'll wait for the connection to fully establish in the first write to it.
            async let asyncLinkedTokenPair = self.requestMetrics.observeAuthTokenFetch {
                try await self.loadLinkedTokenPair()
            }

            // load cached attestations and add them to the sequence of attestations that we should try.
            let cachedNodes = await self.requestMetrics.observeLoadAttestationsFromCache {
                await self.loadCachedAttestations()
            }
            let (unverifiedNodeStream, unverifiedNodeContinuation) = AsyncStream.makeStream(of: ValidatedAttestationOrAttestation.self)
            for cachedAttestation in cachedNodes {
                unverifiedNodeContinuation.yield(cachedAttestation)
            }

            let nodeSelectedEvent = TC2Event<Void>()
            let ropesInvokeRequestSentEvent = TC2Event<Void>()
            let linkedTokenPair = try await asyncLinkedTokenPair

            let result = await withTaskGroup(of: RunSubTask.self, returning: Result<Void, any Error>.self) { taskGroup in
                self.logger.log("Entered main task group")

                // 1. ropes connection
                taskGroup.addTask {
                    return .ropesRequestDidFinish(
                        await Result {
                            let requestHeaders = try self.makeRopesRequestHeaders(
                                token: linkedTokenPair,
                                sessionCount: sessionCount,
                                cachedAttestations: cachedNodes
                            )

                            try await self.runRopesRequest(
                                requestHeaders: requestHeaders,
                                cachedNodes: cachedNodes,
                                inbound: inbound,
                                outbound: outbound,
                                unverifiedNodeContinuation: unverifiedNodeContinuation,
                                ropesInvokeRequestSentEvent: ropesInvokeRequestSentEvent,
                                nodeSelectedEvent: nodeSelectedEvent
                            )
                        }
                    )
                }

                // 2. data stream
                taskGroup.addTask {
                    return .dataSubStreamDidFinished(
                        await Result {
                            try await ropesInvokeRequestSentEvent()
                            try await ohttpStreams.withOHTTPSubStream(
                                ohttpContext: self.requestOHTTPContext,
                                standaloneAEADKey: self.configuration.aeadKey
                            ) { dataStreamInbound, dataStreamOutbound in
                                try await self.sendLoop(
                                    dataStreamOutbound: dataStreamOutbound,
                                    nodeSelectedEvent: nodeSelectedEvent,
                                    linkedTokenPair: linkedTokenPair
                                )
                            }
                        }
                    )
                }

                // 3. node streams
                taskGroup.addTask {
                    return .nodeSubStreamsDidFinished(
                        await Result {
                            try await self.runNodeStreams(
                                unverifiedNodeStream,
                                ropesInvokeRequestSentEvent: ropesInvokeRequestSentEvent,
                                ohttpStreamFactory: ohttpStreams
                            )
                        }
                    )
                }

                // 4. connection metrics
                if let metricsReporter = ohttpStreams as? NWConnectionEstablishmentReportProvider {
                    taskGroup.addTask {
                        do {
                            try await metricsReporter.connectionReady
                            self.requestMetrics.reportConnectionReady()
                            let establishReport = try await metricsReporter.connectionEstablishReport
                            self.requestMetrics.reportConnectionEstablishReport(establishReport)
                        } catch {
                            self.requestMetrics.reportConnectionError(error)
                        }
                        return .connectionMetricsReportingFinished
                    }
                }

                var nodeStreamsError: (any Error)?
                var dataStreamError: (any Error)?
                var ropesError: (any Error)?

                // if one subtask fails, we need to stop all the other ones by throwing
                while let nextResult = await taskGroup.next() {
                    switch nextResult {
                    case .ropesRequestDidFinish(.success):
                        self.logger.log("\(self.lp) Ropes request finished successfully")
                    // We MUST NOT cancel the taskGroup here! Background:
                    //
                    // `ropesRequestDidFinish` means that we received the trailers from ROPES,
                    // as all node messages are proxied through ROPES, this also means that we
                    // won't receive any further messages in the data or nodes streams.
                    // HOWEVER as we use different tasks for processing the different ohttp-
                    // substreams those messages may not be consumed yet because of different
                    // task schedulings! Cancellation of the still running tasks may lead to
                    // truncation of the response. Cancellation is not necessary anyway, since
                    // the streams have already finished by definition (the ROPES request has
                    // finished).

                    case .ropesRequestDidFinish(.failure(let failure)):
                        self.logger.log("\(self.lp) Ropes request failed. Error: \(failure)")
                        self.logger.debug("\(self.lp) Cancelling main task group")
                        taskGroup.cancelAll()
                        ropesError = failure

                    case .dataSubStreamDidFinished(.success):
                        self.logger.log("\(self.lp) Data substream task finished successfully")
                        break

                    case .dataSubStreamDidFinished(.failure(let failure)):
                        self.logger.log("\(self.lp) Data substream task failed. Error: \(failure)")
                        dataStreamError = failure

                    case .nodeSubStreamsDidFinished(.success):
                        self.logger.log("\(self.lp) Node substreams task finished successfully")
                        break

                    case .nodeSubStreamsDidFinished(.failure(let failure)):
                        if let structuredError = failure as? TrustedRequestError,
                            structuredError.code == .failedToValidateAllAttestations
                        {
                            // if all attestations are invalid, there is no further progress that
                            // we can make!
                            taskGroup.cancelAll()
                        }
                        self.logger.log("\(self.lp) Node substreams task failed. error: \(failure)")
                        nodeStreamsError = failure

                    case .connectionMetricsReportingFinished:
                        self.logger.log("\(self.lp) Connection metrics reporting finished")
                    }
                }

                // If we have a `failedToValidateAllAttestations` error from the nodeSubStream task,
                // we should use this error. In all other cases we are interested in what ROPES
                // tells us, as this gives the best signal of what went wrong. But of course only if
                // ropesError is not a cancellation error.
                //
                // We return a Result<Void, any Error> from here, as nonThrowingTaskGroups can
                // currently not throw.
                if let structuredError = nodeStreamsError as? TrustedRequestError,
                    structuredError.code == .failedToValidateAllAttestations
                {
                    return .failure(structuredError)
                }
                if let ropesError, ropesError as? CancellationError == nil {
                    return .failure(ropesError)
                }
                if let nodeStreamsError, nodeStreamsError as? CancellationError == nil {
                    return .failure(nodeStreamsError)
                }
                if let dataStreamError, dataStreamError as? CancellationError == nil {
                    return .failure(dataStreamError)
                }
                if ropesError != nil || nodeStreamsError != nil || dataStreamError != nil {
                    return .failure(CancellationError())
                }
                return .success(())
            }

            try result.get()
        }
    }

    private func checkRateLimiting() async throws -> UInt {
        let requestMetadataForRateLimit = RateLimiterRequestMetadata(
            configuration: self.configuration,
            paramaters: self.parameters
        )
        if let rateLimitInfo = await self.rateLimiter.rateLimitDenialInfo(now: Date.now, for: requestMetadataForRateLimit, sessionID: configuration.sessionID) {
            // This means the rate limiter does not want us to proceed.
            throw TrustedCloudComputeError.deniedDueToRateLimit(rateLimitInfo: rateLimitInfo)
        }

        let sessionCount: UInt
        if let sessionID = self.configuration.sessionID {
            sessionCount = await self.rateLimiter.sessionProgress(now: Date.now, for: sessionID)
            self.logger.log("\(self.lp): using session identifier \(sessionID) with progress \(sessionCount)")
        } else {
            sessionCount = 0
            self.logger.log("\(self.lp): no session identifier on request")
        }
        return sessionCount
    }

    private func updateRateLimiting() async {
        let requestMetadataForRateLimit = RateLimiterRequestMetadata(
            configuration: self.configuration,
            paramaters: self.parameters
        )
        // This is sent as we run the ropes request outbound send. It
        // is positioned this way so that if any of the non-ropes request
        // work fails, the rate limiter is not charged. But we want to
        // be certain that if there is a possibility ropes sees the
        // outbound request, that we have tracked it.
        await self.rateLimiter.appendSuccessfulRequest(requestMetadata: requestMetadataForRateLimit, sessionID: configuration.sessionID, timestamp: Date.now)
    }

    struct LinkedTokenPair {
        var tokenGrantingToken: Data
        var ott: Data
        var salt: Data
    }

    private func loadLinkedTokenPair() async throws -> LinkedTokenPair {
        // Get the token granting token
        let linkedtokenPair: (Data, Data, Data)?
        do {
            linkedtokenPair = try await self.tokenProvider.requestToken()
        } catch {
            throw TrustedRequestError(code: .failedToFetchPrivateAccessTokens, underlying: [error])
        }

        guard let linkedtokenPair else {
            throw TrustedRequestError(code: .failedToFetchPrivateAccessTokens)
        }

        return LinkedTokenPair(
            tokenGrantingToken: linkedtokenPair.0,
            ott: linkedtokenPair.1,
            salt: linkedtokenPair.2
        )
    }

    private func loadCachedAttestations() async -> [ValidatedAttestationOrAttestation] {
        // without store we can't load any attestations
        guard let store = self.attestationStore else {
            self.logger.error("\(self.lp): unable to access attestation store")
            return []
        }
        // get all unexpired attestations
        guard let prefetchParameters = TC2PrefetchParameters().prefetchParameters(invokeParameters: self.parameters) else {
            self.logger.error("\(self.lp): invalid set of parameters for prefetching")
            return []
        }

        let maxPrefetchedAttestations = self.configuration.maxPrefetchedAttestations
        let cachedAttestations = await store.getAttestationsForRequest(parameters: prefetchParameters, serverRequestID: self.serverRequestID, maxAttestations: maxPrefetchedAttestations)
        self.logger.log("\(self.lp): Total cached attestations from store: \(cachedAttestations.count) maxPrefetchedAttestations: \(maxPrefetchedAttestations)")

        var count = 0
        let result = cachedAttestations.map { (key, validatedAttestation) in
            defer {
                count += 1
            }

            self.logger.log("\(self.lp): creating verified node with identifier: \(key), ohttpcontext: \(count + 10)")
            return ValidatedAttestationOrAttestation.cachedValidatedAttestation(validatedAttestation, ohttpContext: UInt64(count + 10))
        }
        return result
    }

    // MARK: Ropes request

    private func runRopesRequest(
        requestHeaders: HTTPFields,
        cachedNodes: [ValidatedAttestationOrAttestation],
        inbound: ConnectionFactory.Inbound,
        outbound: ConnectionFactory.Outbound,
        unverifiedNodeContinuation: AsyncStream<ValidatedAttestationOrAttestation>.Continuation,
        ropesInvokeRequestSentEvent: TC2Event<Void>,
        nodeSelectedEvent: TC2Event<Void>
    ) async throws {
        defer { self.logger.debug("\(self.lp) Finished root connection subtask") }

        let httpRequest = HTTPRequest(
            method: .post,
            scheme: "https",
            authority: self.configuration.trustedRequestHostname,
            path: self.configuration.trustedRequestPath,
            headerFields: requestHeaders
        )

        let invokeRequestMessage = self.makeInvokeRequest(cachedNodes: cachedNodes)
        let framer = Framer()
        let invokeRequestPayload = try framer.frameMessage(invokeRequestMessage)

        await self.updateRateLimiting()

        try await self.requestMetrics.observeSendingRopesRequest {
            // the timeout around sending the invoke request, is used as a connection establish
            // timeout, since NW will automatically retry to create a connection, if it could not
            // establish one at first try.
            try await withCancellationAfterTimeout(duration: .seconds(10), clock: self.clock) {
                try await outbound.write(
                    content: invokeRequestPayload,
                    contentContext: .init(request: httpRequest),
                    isComplete: true
                )
            }
        }

        ropesInvokeRequestSentEvent.fireNonisolated()

        // NOTE: Within the Swift 6 timeframe swift should learn, that we have moved
        //       inbound.
        try await self.handleRopesConnectionResponses(
            inbound: inbound,
            unverifiedNodeContinuation: unverifiedNodeContinuation,
            nodeSelectedEvent: nodeSelectedEvent
        )
    }

    private func filter(workloadParameters params: [String: String]) -> [String: String] {
        let allowed = self.configuration.allowedWorkloadParameters
        return params.filter { elm in
            if allowed.contains(elm.key) {
                return true
            } else {
                self.logger.warning("\(self.lp) found workload parameter not in allow list: \(elm.key)")
                return false
            }
        }
    }

    private func makeRopesRequestHeaders(
        token: LinkedTokenPair,
        sessionCount: UInt,
        cachedAttestations: [ValidatedAttestationOrAttestation]
    ) throws -> HTTPFields {
        let filteredWorkloadParameters = filter(workloadParameters: parameters.pipelineArguments)
        let workloadParametersAsJSON = try self.jsonEncoder.encode(filteredWorkloadParameters)
        let workloadParametersAsString = String(data: workloadParametersAsJSON, encoding: .utf8) ?? ""

        var headers: HTTPFields = [
            .appleRequestUUID: self.serverRequestID.uuidString,
            .appleClientInfo: tc2OSInfo,
            .appleWorkload: parameters.pipelineKind,
            .appleWorkloadParameters: workloadParametersAsString,
            .appleQOS: self.configuration.serverQoS.rawValue,
            .appleBundleID: self.configuration.bundleID,
            .appleSessionProgress: String(sessionCount),
            .contentType: HTTPField.Constants.contentTypeMessageRopesRequest,
            .userAgent: HTTPField.Constants.userAgentTrustedCloudComputeD,
            .authorization: "PrivateToken token=\"\(base64URL(token.ott))\"",
        ]
        if let featureIdentifier = self.configuration.featureID {
            headers[.appleFeatureID] = featureIdentifier
        }
        if let automatedDeviceGroup = OSASystemConfiguration.automatedDeviceGroup() {
            headers[.appleAutomatedDeviceGroup] = automatedDeviceGroup
        }
        if let testSignalHeader = self.configuration.testSignalHeader {
            headers[.appleTestSignal] = testSignalHeader
        }
        if let testOptionsHeader = self.configuration.testOptionsHeader {
            headers[.appleTestOptions] = testOptionsHeader
        }

        if let overrideCellID = self.configuration.overrideCellID {
            // If there is an override cell id, we ALWAYS want to set it, regardless of presence of cached attestations
            headers[.appleServerHint] = overrideCellID

            // We also set a flag to mark that there is an overriden cell id
            headers[.appleServerHintForReal] = "true"
        } else if let node = cachedAttestations.first, let cellID = node.validatedCellID {
            headers[.appleServerHint] = cellID
        }

        // The Authorization header is too big, it blows away other values on the log line.
        var loggedHeaders = headers
        if let authString = headers[HTTPField.Name.authorization] {
            loggedHeaders[.authorization] = authString.prefix(32) + "<...>"
        }
        self.logger.log("\(self.lp) sending headers: \(String(describing: loggedHeaders))")
        return headers
    }

    private func makeInvokeRequest(cachedNodes: [ValidatedAttestationOrAttestation]) -> Proto_Ropes_HttpService_InvokeRequest {
        Proto_Ropes_HttpService_InvokeRequest.with { req in
            req.type = .setupRequest(
                .with({ setupRequest in
                    setupRequest.encryptedRequestOhttpContext = UInt32(self.requestOHTTPContext)
                    setupRequest.capabilities = .with { caps in
                        caps.compressionAlgorithm = [.brotli]
                    }
                    setupRequest.attestationMappings = cachedNodes.map { (node) in
                        Proto_Ropes_HttpService_InvokeRequest.SetupRequest.AttestationMapping.with {
                            self.logger.log("\(self.lp): adding prefetched attestation for node: \(node.identifier) ohttpContext: \(UInt32(node.ohttpContext))")
                            $0.nodeIdentifier = node.identifier
                            $0.ohttpContext = UInt32(node.ohttpContext)
                        }
                    }
                }))
        }
    }

    private func makePrivateCloudComputeSendAuthTokenRequest(
        _ token: LinkedTokenPair
    ) -> Proto_PrivateCloudCompute_PrivateCloudComputeRequest {
        Proto_PrivateCloudCompute_PrivateCloudComputeRequest.with {
            $0.type = .authToken(
                .with { at in
                    at.tokenGrantingToken = token.tokenGrantingToken
                    at.ottSalt = token.salt
                }
            )
        }
    }

    private func makePrivateCloudComputeSendApplicationPayloadRequest(
        data: Data
    ) -> Proto_PrivateCloudCompute_PrivateCloudComputeRequest {
        Proto_PrivateCloudCompute_PrivateCloudComputeRequest.with {
            $0.type = .applicationPayload(data)
        }
    }

    private func handleRopesConnectionResponses(
        inbound: ConnectionFactory.Inbound,
        unverifiedNodeContinuation: AsyncStream<ValidatedAttestationOrAttestation>.Continuation,
        nodeSelectedEvent: TC2Event<Void>
    ) async throws {
        let responseMessageStream =
            inbound
            .compactMap { try self.processResponseContext($0) }
            .deframed(lengthType: UInt32.self, messageType: Proto_Ropes_HttpService_InvokeResponse.self)

        do {
            for try await message in responseMessageStream {
                self.logger.log("\(self.lp): received message: \(String(describing: message.type))")

                switch message.type {
                case .attestationList(let attestationList):
                    self.handleAttestationList(attestationList, continuation: unverifiedNodeContinuation)

                case .nodeSelected:
                    self.requestMetrics.nodeSelected()
                    nodeSelectedEvent.fireNonisolated()

                case .loggingMetadata(let metadata):
                    self.logger.log("\(self.lp): logging metadata message: \(metadata.message)")

                case .rateLimitConfigurationList(let rateLimitConfigurationList):
                    self.logger.debug("\(self.lp): received \(rateLimitConfigurationList.rateLimitConfiguration.count) rate limit configurations")
                    self.handleRateLimitConfigurationList(rateLimitConfigurationList)

                case .expiredAttestationList(let expiredAttestationList):
                    self.logger.debug("\(self.lp): received expired attestation message for paramaters  \(String(describing: self.parameters)). Will refresh attestations out of band")
                    self.eventStreamContinuation.yield(.expiredAttestationList(expiredAttestationList, self.parameters))

                case .noFurtherAttestations:
                    // ROPES sends a new message no_further_attestations in the cases:
                    //  (1) cache miss, send the message after sending attestation_list
                    //  (2) cache hit, send the message before sending node_selected.
                    // This means we can use this as the signal that we won't receive any
                    // further attestations from ROPES inside this trusted request.
                    unverifiedNodeContinuation.finish()
                    self.requestMetrics.noFurtherAttestations()

                case .compressedAttestationList(let compressedAttestationList):
                    let compressedAttestations = tc2AttestationListFromCompressedAttestationList(compressedAttestationList, logger: self.logger)
                    self.handleAttestationList(compressedAttestations, continuation: unverifiedNodeContinuation)

                case .none:
                    break

                @unknown default:
                    self.logger.warning("\(self.lp): unknown: \(String(describing: message.type))")
                }
            }
        } catch {
            nodeSelectedEvent.fireNonisolated(throwing: error)
            throw error
        }
    }

    private func processResponseContext(_ received: NWConnectionReceived) throws -> Data? {
        self.logger.log("\(self.lp): receive: content: \(String(describing: received.data)), contentContext: \(String(describing: received.contentContext)), isComplete: \(received.isComplete)")

        // the response head and response end are hidden in the contentContext. If there is no
        // contextContext we just forward the data for further processing.
        guard let responseContext = received.contentContext, let httpResponse = responseContext.httpResponse else {
            return received.data
        }

        for field in httpResponse.headerFields {
            self.logger.log("\(self.lp): Ropes response header: \(field.name): \(field.value)")
        }

        if let httpMetadata = responseContext.protocolMetadata(definition: NWProtocolHTTP.definition) as? NWProtocolHTTP.Metadata {
            if let trailerFields = httpMetadata.trailerFields {
                for field in trailerFields {
                    self.logger.log("\(self.lp): Ropes response trailer: \(field.name): \(field.value)")
                }
            }
        }

        if received.isComplete {
            try self.processRopesResponseEnd(httpResponse, contentContext: responseContext)
        } else {
            self.requestMetrics.ropesConnectionResponseReceived(httpResponse)
            try self.processRopesResponseHead(httpResponse, contentContext: responseContext)
        }

        return received.data
    }

    // This will produce a rate limit filter that is narrowly targeted at
    // requests "of this type," more or less meaning that other requests
    // should not be impacted by a retry-after response; only substantially
    // similar requests. This is a spec change, in response to:
    // rdar://128609738 (CARRY 22 (in 24h) unexpected errors "DeniedDueToRateLimit: a rate limit of zero is in place for requests of this type")
    private func specificRateLimitFilter() -> RateLimitFilter {
        let bundleID = self.configuration.bundleID
        let featureID = self.configuration.featureID
        let workloadType = self.parameters.pipelineKind
        let workloadParams = self.filter(workloadParameters: self.parameters.pipelineArguments)
        return RateLimitFilter(bundleID: bundleID, featureID: featureID, workloadType: workloadType, workloadParams: workloadParams)
    }

    private func processRopesResponseHead(_ response: HTTPResponse, contentContext: NWConnection.ContentContext) throws {
        // If it is a bad request, ROPES will fail it with a non-200 response and send errors in headers

        let responseMetadata = TC2RopesResponseMetadata(response, contentContext: contentContext)
        if responseMetadata.isAvailabilityConcern, let retryAfter = responseMetadata.retryAfter {
            let rateLimitConfig = RateLimitConfiguration(
                filter: self.specificRateLimitFilter(),
                timing: .init(
                    now: Date.now,
                    retryAfter: retryAfter,
                    config: self.configuration
                )
            )
            self.eventStreamContinuation.yield(.rateLimitConfigurations([rateLimitConfig]))
        }

        if responseMetadata.code != .ok {
            throw TrustedCloudComputeError(responseMetadata: responseMetadata)
        }
    }

    private func processRopesResponseEnd(_ response: HTTPResponse, contentContext: NWConnection.ContentContext) throws {
        // If it is a bad request, ROPES will fail it with a non-200 response and send errors in headers
        // If for whatever reason, after sending the initial 200 OK, there is an error, ROPES indicates that in trailers
        // Responses/Trailers from ROPES can contain these headers:
        //  “status” response header contains the gRPC status code of the error
        //  “error-code” response header contains ropes-defined error codes
        //  “description” response header contains the description of the error. This header might not be set in production environments
        //  “cause” response header contains a short description of the cause of the error. This header might not be set in production environments
        //  “retry-after” response headers contains the number of seconds that the client should wait before retrying
        //  "ttr-*" response headers contains the context of a Tap-to-radar indicaiton from ROPES

        let responseMetadata = TC2RopesResponseMetadata(response, contentContext: contentContext)
        if responseMetadata.isAvailabilityConcern, let retryAfter = responseMetadata.retryAfter {
            let rateLimitConfig = RateLimitConfiguration(
                filter: self.specificRateLimitFilter(),
                timing: .init(
                    now: Date.now,
                    retryAfter: retryAfter,
                    config: self.configuration
                )
            )
            self.eventStreamContinuation.yield(.rateLimitConfigurations([rateLimitConfig]))
        }

        #if os(iOS)
        if let ttrTitle = responseMetadata.ttrTitle, self.configuration.environment != TC2EnvironmentNames.production.rawValue {
            // contains ttr title, meaning that server wants us to prompt ttr
            let ttrContext = TapToRadarContext(
                ttrTitle: ttrTitle,
                ttrDescription: responseMetadata.ttrDescription,
                ttrComponentID: responseMetadata.ttrComponentID,
                ttrComponentName: responseMetadata.ttrComponentName,
                ttrComponentVersion: responseMetadata.ttrComponentVersion
            )
            self.eventStreamContinuation.yield(.tapToRadarIndicationReceived(context: ttrContext))
        }
        #endif

        // check that ROPES didn't mark the request as failed in the trailers.
        if responseMetadata.status != .ok || responseMetadata.receivedErrorCode != .errorCode(.success) {
            throw TrustedCloudComputeError(responseMetadata: responseMetadata)
        }
    }

    private func handleAttestationList(
        _ attestationList: Proto_Ropes_Common_AttestationList,
        continuation: AsyncStream<ValidatedAttestationOrAttestation>.Continuation
    ) {
        self.logger.debug("\(self.lp): received more \(attestationList.attestation.count) attestations")

        let attestations = attestationList.attestation.map { attestation in
            self.logger.debug("\(self.lp): attestation with \(attestation.ohttpContext) ohttp context")
            return ValidatedAttestationOrAttestation.inlineAttestation(
                Attestation(
                    attestation: attestation,
                    requestParameters: self.parameters
                ),
                ohttpContext: UInt64(attestation.ohttpContext)
            )
        }

        self.requestMetrics.attestationsReceived(attestations)

        for attestation in attestations {
            continuation.yield(attestation)
        }

        // NOTE: do not cache attestation. See: rdar://124965521 (Attestations received in response to an invoke should not be cached)
        // Attestations should only be added as a result of prefetching.
    }

    private func handleRateLimitConfigurationList(
        _ rateLimitConfigs: Proto_Ropes_RateLimit_RateLimitConfigurationList
    ) {
        let list = rateLimitConfigs.rateLimitConfiguration.compactMap { proto in
            if let rateLimitConfig = RateLimitConfiguration(
                now: Date.now,
                proto: proto,
                config: self.configuration
            ) {
                return rateLimitConfig
            } else {
                self.logger.error("\(self.lp) unable to process rate limit configuration \(String(describing: proto))")
                return nil
            }
        }
        self.eventStreamContinuation.yield(.rateLimitConfigurations(list))
    }

    // MARK: Send data stream

    private func sendLoop(
        dataStreamOutbound: ConnectionFactory.OHTTPSubStreamFactory.Outbound,
        nodeSelectedEvent: TC2Event<Void>,
        linkedTokenPair: LinkedTokenPair
    ) async throws {
        var nodeSelected: Bool = false
        var budget = TrustedRequestConstants.maxDataToSendBeforeNodeSelected
        let framer = Framer()

        try await self.requestMetrics.observeAuthTokenSend {
            let authMessage = self.makePrivateCloudComputeSendAuthTokenRequest(linkedTokenPair)
            let authFrame = try framer.frameMessage(authMessage)
            budget -= authFrame.count

            self.logger.debug("\(self.lp) Sending auth message on data stream. Remaining budget before node selected: \(budget)")

            try await dataStreamOutbound.write(
                content: authFrame,
                contentContext: .defaultMessage,
                isComplete: false
            )
        }

        var userStreamIsFinished = false
        while !userStreamIsFinished {
            try await self.outgoingUserDataWriter.withNextOutgoingElement { outgoingUserData in
                self.logger.debug("\(self.lp) Received user data to forward to server")
                if outgoingUserData.isComplete {
                    userStreamIsFinished = true
                    if outgoingUserData.data.isEmpty {
                        // if we are in the last message and don't need to transfer any more data,
                        // we must not wrap the nothing data in a payload request proto. Instead we
                        // only need to signal to the network stack, that write has completed.
                        return try await dataStreamOutbound.write(
                            content: nil,
                            contentContext: .defaultMessage,
                            isComplete: true
                        )
                    }
                }
                let outgoingMessage = self.makePrivateCloudComputeSendApplicationPayloadRequest(
                    data: outgoingUserData.data
                )

                let message = try framer.frameMessage(outgoingMessage)
                self.requestMetrics.receivedOutgoingUserDataChunk()

                if nodeSelected {
                    self.logger.debug("\(self.lp) Sending message on data stream, node selected")
                    try await self.requestMetrics.observeDataWrite(bytesToSend: message.count) {
                        try await dataStreamOutbound.write(
                            content: message,
                            contentContext: .defaultMessage,
                            isComplete: outgoingUserData.isComplete
                        )
                    }
                } else {
                    if message.count <= budget {  // do we fit into our budget
                        self.logger.debug("\(self.lp) Sending message on data stream, within inital budget: \(budget)")
                        try await self.requestMetrics.observeDataWrite(bytesToSend: message.count, inBudget: true) {
                            try await dataStreamOutbound.write(
                                content: message,
                                contentContext: .defaultMessage,
                                isComplete: outgoingUserData.isComplete
                            )
                        }
                        budget -= message.count
                    } else {
                        self.logger.debug("\(self.lp) Sending message on data stream (\(message.count) bytes), above initial budget: \(budget) bytes")
                        let sendAfterNodeSelected: Data
                        // we are over budget
                        if budget > 0 {
                            // send what we can
                            let prefix = message.prefix(budget)
                            try await self.requestMetrics.observeDataWrite(bytesToSend: prefix.count, inBudget: false) {
                                try await dataStreamOutbound.write(
                                    content: prefix,
                                    contentContext: .defaultMessage,
                                    isComplete: false
                                )
                            }
                            sendAfterNodeSelected = message.dropFirst(prefix.count)
                        } else {
                            sendAfterNodeSelected = message
                        }

                        self.logger.debug("\(self.lp) Waiting on node selected")
                        try await nodeSelectedEvent()
                        nodeSelected = true
                        self.logger.debug("\(self.lp) Node selected")

                        try await self.requestMetrics.observeDataWrite(
                            bytesToSend: sendAfterNodeSelected.count,
                            inBudget: false
                        ) {
                            try await dataStreamOutbound.write(
                                content: sendAfterNodeSelected,
                                contentContext: .defaultMessage,
                                isComplete: outgoingUserData.isComplete
                            )
                        }
                    }
                }
            }
        }
        self.logger.log("\(self.lp) Finished sending all user data")
        self.requestMetrics.dataStreamFinished()
    }

    // MARK: Node streams

    private enum RunNodeStreamResult {
        case finished
        case verifyAttestationFailed(any Error)
        case cancelledAsOtherNodeSelected
    }

    private func runNodeStreams(
        _ maybeUnverifiedNodeStream: AsyncStream<ValidatedAttestationOrAttestation>,
        ropesInvokeRequestSentEvent: TC2Event<Void>,
        ohttpStreamFactory: ConnectionFactory.OHTTPSubStreamFactory
    ) async throws {
        defer { self.logger.debug("\(self.lp) Leaving runNodesStreams") }

        let (reportableNodeUniqueIdStream, reportableNodeUniqueIdContinuation) = AsyncStream.makeStream(of: String.self)
        async let _: Void = {
            // Here we consume the hardware identifiers of attestations as they are
            // validated by the node streams in their separate tasks. When the tasks
            // are concluded and we know that no further validation will happen, the
            // stream finishes and we publish to the event stream that tracks the
            // distribution of nodes. See rdar://135384108 for details.
            var reportableNodeUniqueIds: [String] = []
            reportableNodeUniqueIds.reserveCapacity(32)
            for await nodeUniqueId in reportableNodeUniqueIdStream {
                reportableNodeUniqueIds.append(nodeUniqueId)
            }
            self.requestMetrics.inlineAttestationsValidated(reportableNodeUniqueIds)
        }()
        defer { reportableNodeUniqueIdContinuation.finish() }

        try await withThrowingTaskGroup(of: RunNodeStreamResult.self, returning: Void.self) { taskGroup in
            let nodeStreamController = NodeStreamController()

            var runningNodeIDs: Set<String> = []
            var inlineNodeIDs: Set<String> = []
            runningNodeIDs.reserveCapacity(32)  // skip first reallocs
            inlineNodeIDs.reserveCapacity(32)

            for await node in maybeUnverifiedNodeStream {
                if case .inlineAttestation = node {
                    if inlineNodeIDs.count >= self.configuration.maxInlineAttestations {
                        let count = inlineNodeIDs.count
                        // TODO: consider storing this attestation somewhere so we can log it in thtool requests, etc.; probably overkill though.
                        self.logger.warning("\(self.lp): ignoring node \(node.identifier); already have \(count) attestations out of \(self.configuration.maxTotalAttestations) max")
                        continue
                    }

                    inlineNodeIDs.insert(node.identifier)
                }

                if runningNodeIDs.contains(node.identifier) {
                    self.logger.debug("\(self.lp): already have a node with identifier \(node.identifier), conflict: \(node.ohttpContext)")
                    continue
                }

                if runningNodeIDs.count >= self.configuration.maxTotalAttestations {
                    let count = runningNodeIDs.count
                    // TODO: consider storing this attestation somewhere so we can log it in thtool requests, etc.; probably overkill though.
                    self.logger.warning("\(self.lp): ignoring node \(node.identifier); already have \(count) attestations out of \(self.configuration.maxTotalAttestations) max")
                    continue
                }

                runningNodeIDs.insert(node.identifier)

                taskGroup.addTask {
                    self.logger.log("\(self.lp): Creating node stream subtask for node: \(node.identifier)")

                    defer { self.logger.debug("\(self.lp) Leaving node stream subtask for node: \(node.identifier)") }
                    return try await withThrowingTaskGroup(of: RunNodeStreamResult.self, returning: RunNodeStreamResult.self) { taskGroup in
                        taskGroup.addTask {
                            do {
                                try await nodeStreamController.registerNodeStream(nodeID: node.identifier)
                                return .finished
                            } catch {
                                self.logger.debug("\(self.lp): cancelled node stream \(node.identifier)")
                                return .cancelledAsOtherNodeSelected
                            }
                        }

                        taskGroup.addTask {
                            let validatedAttestation: ValidatedAttestation
                            let ohhtpContext: UInt64 = node.ohttpContext
                            switch node {
                            case .cachedValidatedAttestation(let attestation, _):
                                validatedAttestation = attestation

                            case .inlineAttestation(let attestation, _):
                                self.logger.log("\(self.lp): verifying attestation for context \(node.ohttpContext)")
                                // the state after verify attestation is set in `verifyAttestation(node:)`
                                do {
                                    validatedAttestation = try await self.verifyAttestation(attestation: attestation)
                                    if let uniqueNodeIdentifier = validatedAttestation.uniqueNodeIdentifier {
                                        reportableNodeUniqueIdContinuation.yield(uniqueNodeIdentifier)
                                    }
                                } catch {
                                    return .verifyAttestationFailed(error)
                                }
                            }

                            // the sentKey state us set `runEndpointRequest(node:, request:)`
                            return try await self.runNodeRequest(
                                validatedAttestation: validatedAttestation,
                                ohttpContext: ohhtpContext,
                                ropesInvokeRequestSentEvent: ropesInvokeRequestSentEvent,
                                nodeStreamController: nodeStreamController,
                                ohttpStreamFactory: ohttpStreamFactory
                            )
                        }

                        switch await taskGroup.nextResult()! {
                        case .success(let success):
                            taskGroup.cancelAll()
                            return success
                        case .failure(let error):
                            taskGroup.cancelAll()
                            throw error
                        }
                    }
                }  // end node subtask group
            }  // end node for loop

            self.logger.debug("\(self.lp) Not expecting more attestations. Running with \(runningNodeIDs.count) attestations")

            var verificationFailures: [any Error] = []
            var atLeastOneSucceeded = false
            var error: (any Error)?

            var completed = 0
            while let result = await taskGroup.nextResult() {
                completed += 1
                self.logger.debug("\(self.lp) Node substream task finished. Remaining: \(runningNodeIDs.count - completed)")
                switch result {
                case .success(.verifyAttestationFailed(let error)):
                    verificationFailures.append(error)

                case .success(.finished), .success(.cancelledAsOtherNodeSelected):
                    atLeastOneSucceeded = true

                case .failure(let taskError):
                    error = taskError
                }
            }

            self.logger.debug("\(self.lp) All \(runningNodeIDs.count) node substreams have finished")
            defer { self.logger.debug("\(self.lp) Leaving runNodesStreams taskGroup. Success: \(atLeastOneSucceeded)") }

            if !atLeastOneSucceeded {
                if verificationFailures.count == runningNodeIDs.count, verificationFailures.count > 0 {
                    error = TrustedRequestError(
                        code: .failedToValidateAllAttestations,
                        underlying: verificationFailures
                    )
                }

                if let error {
                    self.requestMetrics.nodeResponseStreamsFailed(error)
                    throw error
                }
            }
        }
    }

    private func verifyAttestation(attestation: Attestation) async throws -> ValidatedAttestation {
        try await self.requestMetrics.observeAttestationVerify(
            nodeID: attestation.nodeID
        ) {
            do {
                let validatedAttestation = try await self.attestationVerifier.validate(attestation: attestation)
                self.logger.log("\(self.lp): attestation success with package key \(validatedAttestation.publicKey), validationExpiration: \(validatedAttestation.attestationExpiry)")
                return validatedAttestation
            } catch {
                self.logger.log("\(self.lp): attestation failure with error \(error)")
                throw error
            }
        }
    }

    private func runNodeRequest(
        validatedAttestation: ValidatedAttestation,
        ohttpContext: UInt64,
        ropesInvokeRequestSentEvent: TC2Event<Void>,
        nodeStreamController: NodeStreamController,
        ohttpStreamFactory: ConnectionFactory.OHTTPSubStreamFactory
    ) async throws -> RunNodeStreamResult {
        let nodeID = validatedAttestation.attestation.nodeID
        self.logger.log("\(self.lp): starting node stream to \(nodeID); creating request...")

        return try await ohttpStreamFactory.withOHTTPSubStream(
            ohttpContext: ohttpContext,
            gatewayKeyConfig: validatedAttestation.publicKey,
            mediaType: "application/protobuf"
        ) { nodeInbound, nodeOutbound in
            // we must ensure that the ropes invoke request got send in the http request first.
            try await ropesInvokeRequestSentEvent()

            // we send our aead key first
            try await self.requestMetrics.observeSendingKeyToNode(nodeID: nodeID) {
                try await nodeOutbound.write(
                    content: self.configuration.aeadKey,
                    contentContext: .defaultMessage,
                    isComplete: true
                )
            }

            // now lets consume all responses
            let deframed =
                nodeInbound
                .compactMap { $0.data }
                .deframed(
                    lengthType: UInt32.self,
                    messageType: Proto_PrivateCloudCompute_PrivateCloudComputeResponse.self
                )

            var isFirstMessage = true

            receiveLoop: for try await message in deframed {
                self.logger.debug("\(self.lp): Received message from node \(nodeID): \(String(describing: message.type))")
                if isFirstMessage {
                    self.logger.debug("\(self.lp): Node has received data, cancelling all other node streams")
                    nodeStreamController.dataReceived(nodeID: nodeID)
                    self.requestMetrics.nodeFirstResponseReceived(nodeID: nodeID)
                    isFirstMessage = false
                }

                switch message.type {
                case .responseUuid(let uuidData):
                    guard uuidData.count == 16 else {
                        throw TrustedRequestError(code: .invalidResponseUUID)
                    }

                    let uuid: UUID = uuidData.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
                        var uuid: uuid_t = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                        let bytesCopied = withUnsafeMutableBytes(of: &uuid) { targetPtr in
                            bytes.copyBytes(to: targetPtr)
                        }
                        precondition(bytesCopied == 16)
                        return UUID(uuid: uuid)
                    }

                    self.logger.debug("\(self.lp): Response UUID: \(uuid)")

                case .responseSummary(let responseSummary):
                    self.requestMetrics.nodeSummaryReceived(nodeID: nodeID)
                    self.logger.debug("\(self.lp): Response summary: \(responseSummary.debugDescription)")
                    switch responseSummary.responseStatus {
                    case .ok:
                        break  // no error
                    case .unauthenticated:
                        throw TrustedRequestError(code: .responseSummaryIndicatesUnauthenticated)
                    case .invalidRequest:
                        throw TrustedRequestError(code: .responseSummaryIndicatesInvalidRequest)
                    default:
                        throw TrustedRequestError(code: .responseSummaryIndicatesFailure)
                    }

                case .responsePayload(let payload):
                    self.logger.debug("\(self.lp): Received payload \(payload.count) bytes from node \(ohttpContext)")

                    self.requestMetrics.nodeResponsePayloadReceived(nodeID: nodeID, bytes: payload.count)
                    try await self.incomingUserDataReader.forwardData(payload)

                case .none:
                    // TBD: Should we fail the request here. Not getting a type looks like a
                    //      protocol error from the server.
                    break

                @unknown default:
                    break
                }
            }

            self.requestMetrics.nodeResponseFinished(nodeID: nodeID)
            self.logger.debug("\(self.lp): Received all messages in node stream: \(nodeID)")

            return .finished
        }
    }
}

enum ValidatedAttestationOrAttestation {
    case cachedValidatedAttestation(ValidatedAttestation, ohttpContext: UInt64)
    case inlineAttestation(Attestation, ohttpContext: UInt64)

    var identifier: String {
        switch self {
        case .cachedValidatedAttestation(let validatedAttestation, _):
            return validatedAttestation.attestation.nodeID
        case .inlineAttestation(let attestation, _):
            return attestation.nodeID
        }
    }

    var ohttpContext: UInt64 {
        switch self {
        case .cachedValidatedAttestation(_, let ohttpContext):
            return ohttpContext
        case .inlineAttestation(_, let ohttpContext):
            return ohttpContext
        }
    }

    var maybeValidatedCellID: String {
        switch self {
        case .cachedValidatedAttestation(let validatedAttestation, _):
            return validatedAttestation.validatedCellID ?? ""
        case .inlineAttestation(let attestation, _):
            return "unvalidated(\(attestation.unvalidatedCellID ?? ""))"
        }
    }

    var validatedCellID: String? {
        switch self {
        case .cachedValidatedAttestation(let validatedAttestation, _):
            return validatedAttestation.validatedCellID
        default:
            return nil
        }
    }

    var cloudOSVersion: String {
        switch self {
        case .cachedValidatedAttestation(let validatedAttestation, _):
            return validatedAttestation.attestation.cloudOSVersion
        case .inlineAttestation(let attestation, _):
            return attestation.cloudOSVersion
        }
    }

    var cloudOSReleaseType: String {
        switch self {
        case .cachedValidatedAttestation(let validatedAttestation, _):
            return validatedAttestation.attestation.cloudOSReleaseType
        case .inlineAttestation(let attestation, _):
            return attestation.cloudOSReleaseType
        }
    }
}

struct LogPrefix: CustomStringConvertible {
    let requestID: UUID

    var description: String {
        "Request: \(self.requestID)"
    }
}

extension Result {
    init(asyncCatching: () async throws(Failure) -> Success) async {
        do {
            self = try await .success(asyncCatching())
        } catch {
            self = .failure(error)
        }
    }
}

extension TC2RopesResponseMetadata {
    init(_ response: HTTPResponse, contentContext: NWConnection.ContentContext) {
        self = TC2RopesResponseMetadata(code: response.status.code)
        for field in response.headerFields {
            self.set(value: field.value, for: field.name.rawName)
        }

        // If we have trailers we need to consume those as well.
        if let httpMetadata = contentContext.protocolMetadata(definition: NWProtocolHTTP.definition) as? NWProtocolHTTP.Metadata {
            if let trailerFields = httpMetadata.trailerFields {
                for field in trailerFields {
                    self.set(value: field.value, for: field.name.rawName)
                }
            }
        }
    }
}

private func base64URL(_ data: Data) -> String {
    return data.base64EncodedString()
        .replacingOccurrences(of: "+", with: "-")
        .replacingOccurrences(of: "/", with: "_")
        .replacingOccurrences(of: "=", with: "")
}

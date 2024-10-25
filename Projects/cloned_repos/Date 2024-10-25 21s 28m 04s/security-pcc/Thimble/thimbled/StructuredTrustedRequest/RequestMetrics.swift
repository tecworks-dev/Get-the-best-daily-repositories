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
//  RequestMetrics.swift
//  PrivateCloudCompute
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import CloudAttestation
import CloudTelemetry
import Foundation
import GenerativeFunctionsInstrumentation
@_spi(Restricted) import IntelligencePlatformLibrary
@_spi(HTTP) @_spi(NWActivity) import Network
import OSLog
import PrivateCloudCompute
@preconcurrency import os

final class RequestMetrics<
    Clock: _Concurrency.Clock,
    AttestationStore: TC2AttestationStoreProtocol
>: Sendable where Clock.Duration == Duration {
    private struct State: Sendable {
        enum RopesRequestState: Sendable, CustomStringConvertible {
            case initialized
            case waitingForConnection(attestationsActivity: NWActivity)
            case requestSent(attestationsActivity: NWActivity)
            case responseHeadReceived(attestationsActivity: NWActivity)
            case attestationsReceived
            case nodeSelected
            case finished(Duration)
            case failed(TrustedCloudComputeError, Duration)

            var description: String {
                switch self {
                case .initialized:
                    return "Initialized"
                case .requestSent:
                    return "Request sent"
                case .responseHeadReceived:
                    return "Response head received"
                case .attestationsReceived:
                    return "Attestations received"
                case .nodeSelected:
                    return "Node selected"
                case .finished:
                    return "Finished"
                case .failed(let error, _):
                    return "Failed (error: \(error))"
                case .waitingForConnection:
                    return "Waiting for ROPES OHTTP Connection"
                }
            }
        }

        enum DataStreamState: Sendable, CustomStringConvertible {
            case initialized
            case authTokenSent
            case awaitingNodeSelected(bytesSent: Int)
            case nodeSelected(bytesSent: Int)
            case finished(bytesSent: Int)
            case failed(any Error)

            var description: String {
                switch self {
                case .initialized:
                    return "Initialized"
                case .authTokenSent:
                    return "AuthTokenSent"
                case .awaitingNodeSelected(let bytesSent):
                    return "Connected (remaining budget: \(TrustedRequestConstants.maxDataToSendBeforeNodeSelected - bytesSent))"
                case .nodeSelected:
                    return "Node selected"
                case .finished:
                    return "Finished"
                case .failed(let error):
                    return "Failed (error: \(error))"
                }
            }
        }

        enum ResponseStreamState: Sendable, CustomStringConvertible {
            case initialized
            case waitingToSendFirstKey(firstTokenActivity: NWActivity)
            case waitingForNode(firstTokenActivity: NWActivity, interval: OSSignpostIntervalState)
            case receiving(nodeID: String, bytesReceived: Int, interval: OSSignpostIntervalState)
            case finished(nodeID: String, bytesReceived: Int)
            case failed(any Error)

            var description: String {
                switch self {
                case .initialized:
                    return "Initialized"
                case .waitingToSendFirstKey:
                    return "Waiting to send first key"
                case .waitingForNode:
                    return "Waiting for Node"
                case .receiving(let nodeID, _, _):
                    return "Receiving from \(nodeID)"
                case .finished:
                    return "Finished"
                case .failed(let error):
                    return "Failed (error: \(error))"
                }
            }
        }

        enum AuthTokenFetchState: Sendable {
            case notStarted
            case succeeded(Duration)
            case failed(any Error, Duration)
        }

        enum AuthTokenSendState: Sendable {
            case notStarted
            case succeeded(Duration)
            case failed(any Error, Duration)
        }

        enum AttestationsReceivedState: Sendable {
            case noAttestationReceived
            case receivedAttestations(count: Int, durationSinceStart: Duration)
        }

        enum RopesRequestSentState: Sendable {
            case notSend
            case succeeded(Duration)
            case failed(any Error, Duration)
        }

        enum KDataSendState: Sendable {
            case notSend
            /// duration here means the time interval from beginning of the request to the time when we send the first key
            case sent(duration: Duration, count: Int)
        }

        enum FirstChunkSentState: Sendable {
            case notSend
            case succeeded(Duration, withinBudget: Bool)
        }

        enum OHTTPConnectionEstablishmentState {
            case notEstablished
            case established(Duration, NWConnection.EstablishmentReport?)
            case failed(Duration, any Error)
        }

        enum AttestationBundleRef: Equatable {
            case lookupInDatabase
            case data(Data)
        }

        struct NodeMetadata: Sendable {
            enum State: Sendable, CustomStringConvertible {
                case unverified
                case verifying
                case verified
                case verifiedFailed(any Error)
                case sentKey
                case receiving(summaryReceived: Bool, bytesReceived: Int)
                case finished(summaryReceived: Bool, bytesReceived: Int)

                var description: String {
                    switch self {
                    case .unverified:
                        "unverified"
                    case .verifying:
                        "verifying"
                    case .verifiedFailed:
                        "verifiedFailed"
                    case .verified:
                        "verified"
                    case .sentKey:
                        "sentKey"
                    case .receiving:
                        "receiving"
                    case .finished:
                        "finished"
                    }
                }

                var hasReceivedSummary: Bool {
                    switch self {
                    case .unverified, .verifying, .verified, .verifiedFailed, .sentKey:
                        return false
                    case .receiving(let summaryReceived, _),
                        .finished(let summaryReceived, _):
                        return summaryReceived
                    }
                }

                var bytesReceived: UInt64 {
                    switch self {
                    case .unverified, .verifying, .verified, .verifiedFailed, .sentKey:
                        return 0
                    case .receiving(_, let bytesReceived),
                        .finished(_, let bytesReceived):
                        return UInt64(bytesReceived)
                    }
                }
            }

            var state: State
            var nodeID: String
            var attestationBundleRef: AttestationBundleRef
            var ohttpContext: UInt64
            var cloudOSVersion: String?
            var cloudOSReleaseType: String?
            var maybeValidatedCellID: String?
            var ensembleID: String?
            var isFromCache: Bool {
                switch self.attestationBundleRef {
                case .lookupInDatabase:
                    true
                case .data(_):
                    false
                }
            }
        }

        var connectionEstablishState: OHTTPConnectionEstablishmentState = .notEstablished
        var ropesRequestState: RopesRequestState = .initialized
        var ropesRequestSentState: RopesRequestSentState = .notSend
        var dataStreamState: DataStreamState = .initialized
        var responseStreamState: ResponseStreamState = .initialized
        var authTokenFetchState: AuthTokenFetchState = .notStarted
        var authTokenSendState: AuthTokenSendState = .notStarted
        var firstChunkSentState: FirstChunkSentState = .notSend
        var attestationsReceivedState: AttestationsReceivedState = .noAttestationReceived
        var kDataSendState: KDataSendState = .notSend

        var durationSinceStartTillNodeSelected: Duration? = nil
        var durationSinceStartTillLastPayloadChunkSent: Duration? = nil
        var durationSinceStartTillFirstToken: Duration? = nil

        var responseCode: Int?
        var ropesVersion: String?

        var nodes: [String: NodeMetadata] = [:]

        var verifiedAttestationsCount = 0
    }

    private let state = os.OSAllocatedUnfairLock(initialState: State())
    private let clock: Clock

    private let clientRequestID: UUID
    private let serverRequestID: UUID
    private let startDate: Date
    private let startInstant: Clock.Instant
    private let bundleID: String
    private let originatingBundleID: String?
    private let featureID: String?
    private let sessionID: UUID?
    private let environment: String
    private let qos: ServerQoS
    private let parameters: TC2RequestParameters

    private let clientInfo: String
    private let locale: String

    private let logger: Logger
    private let lp: LogPrefix
    private let eventStreamContinuation: AsyncStream<ThimbledEvent>.Continuation

    private let signposter: OSSignposter
    // rdar://126140969 (Consider making OSSignpostID Sendable)
    private let signpostID: OSSignpostID
    private let fullRequestInterval: OSSignpostIntervalState

    /// this is the requestID that will be used for Cloud Telemetry reporting
    /// In PROD, it needs to be different than requestID for privacy concerns
    public let requestIDForEventReporting: UUID
    private let attestationStore: AttestationStore?
    private let encoder = tc2JSONEncoder()

    init(
        clientRequestID: UUID,
        serverRequestID: UUID,
        bundleID: String,
        originatingBundleID: String?,
        featureID: String?,
        sessionID: UUID?,
        environment: String,
        qos: ServerQoS,
        parameters: TC2RequestParameters,
        logger: Logger,
        eventStreamContinuation: AsyncStream<ThimbledEvent>.Continuation,
        clock: Clock,
        store: AttestationStore?
    ) {
        self.clientRequestID = clientRequestID
        self.serverRequestID = serverRequestID
        self.clock = clock
        self.startDate = .now
        self.startInstant = clock.now

        self.bundleID = bundleID
        self.originatingBundleID = originatingBundleID
        self.featureID = featureID
        self.sessionID = sessionID
        self.environment = environment
        self.qos = qos
        self.parameters = parameters

        self.clientInfo = tc2OSInfoWithDeviceModel
        self.locale = Locale.current.identifier

        self.logger = logger
        self.lp = LogPrefix(requestID: serverRequestID)
        self.eventStreamContinuation = eventStreamContinuation
        self.signposter = OSSignposter(logger: self.logger)
        self.signpostID = self.signposter.makeSignpostID()
        self.fullRequestInterval = self.signposter.beginInterval("FullTrustedRequest", id: self.signpostID)

        if self.environment == TC2EnvironmentNames.production.rawValue {
            self.requestIDForEventReporting = UUID()
            self.logger.log("\(self.lp) RequestIDForEventReporting: \(self.requestIDForEventReporting.uuidString)")
        } else {
            self.requestIDForEventReporting = self.serverRequestID
        }

        self.attestationStore = store
    }

    // MARK: - Export

    func makeMetadata() -> TC2TrustedRequestMetadata {
        let state = self.state.withLock { $0 }

        return .init(
            serverRequestID: self.serverRequestID,
            environment: self.environment,
            creationDate: self.startDate,
            bundleIdentifier: self.bundleID,
            featureIdentifier: self.featureID,
            sessionIdentifier: self.sessionID,
            qos: self.qos.rawValue,
            parameters: self.parameters,
            state: "\(state.ropesRequestState)",
            payloadTransportState: "\(state.dataStreamState)",
            responseState: "\(state.responseStreamState)",
            responseCode: state.responseCode,
            ropesVersion: state.ropesVersion,
            endpoints: state.nodes.map { (id, info) in
                TC2TrustedRequestEndpointMetadata(
                    nodeState: "\(info.state)",
                    nodeIdentifier: info.nodeID,
                    ohttpContext: info.ohttpContext,
                    hasReceivedSummary: info.state.hasReceivedSummary,
                    dataReceived: info.state.bytesReceived,
                    cloudOSVersion: info.cloudOSVersion,
                    cloudOSReleaseType: info.cloudOSReleaseType,
                    maybeValidatedCellID: info.maybeValidatedCellID,
                    ensembleID: info.ensembleID,
                    isFromCache: info.isFromCache
                )
            }
        )
    }

    func makeFullRequestMetrics() -> TC2TrustedRequestMetric {
        var requestMetrics = TC2TrustedRequestMetric()
        requestMetrics.bundleID = self.bundleID
        requestMetrics.fields[.eventTime] = .int(Int64(Date().timeIntervalSince1970))
        requestMetrics.fields[.environment] = .string(self.environment)
        requestMetrics.fields[.clientInfo] = .string(self.clientInfo)
        if let featureID = self.featureID {
            requestMetrics.fields[.featureID] = .string(featureID)
        }
        requestMetrics.fields[.bundleID] = .string(self.bundleID)
        if let originatingBundleID = self.originatingBundleID {
            requestMetrics.fields[.originatingBundleID] = .string(originatingBundleID)
        }
        requestMetrics.fields[.locale] = .string(self.locale)
        requestMetrics.fields[.clientRequestid] = .string(self.requestIDForEventReporting.uuidString)

        let stateCopy = self.state.withLock { $0 }

        switch stateCopy.connectionEstablishState {
        case .notEstablished:
            break

        case .established(let duration, let report):
            requestMetrics.fields[.ohttpConnectionEstablishmentSuccess] = true
            requestMetrics.fields[.ohttpConnectionEstablishmentTime] = .milliSeconds(duration)
            if case .url(let proxyURL) = report?.proxyEndpoint {
                requestMetrics.fields[.ohttpProxyVendor] = .string(proxyURL.absoluteString)
            }

        case .failed(let duration, let error):
            requestMetrics.fields[.ohttpConnectionEstablishmentSuccess] = false
            requestMetrics.fields[.ohttpConnectionEstablishmentTime] = .milliSeconds(duration)
            requestMetrics.fields[.ohttpConnectionEstablishmentError] = error.telemetryString
        }

        switch stateCopy.authTokenFetchState {
        case .notStarted:
            break
        case .succeeded(let duration):
            requestMetrics.fields[.authTokenFetchSuccess] = true
            requestMetrics.fields[.authTokenFetchTime] = .milliSeconds(duration)
        case .failed(let error, let duration):
            requestMetrics.fields[.authTokenFetchSuccess] = false
            requestMetrics.fields[.authTokenFetchTime] = .milliSeconds(duration)
            requestMetrics.fields[.authTokenFetchError] = error.telemetryString
        }

        switch stateCopy.authTokenSendState {
        case .notStarted:
            break
        case .succeeded(let duration):
            requestMetrics.fields[.authTokenSendTime] = .milliSeconds(duration)
            requestMetrics.fields[.authTokenSendSuccess] = true
        case .failed(let error, let duration):
            requestMetrics.fields[.authTokenSendTime] = .milliSeconds(duration)
            requestMetrics.fields[.authTokenSendSuccess] = false
            requestMetrics.fields[.authTokenSendError] = error.telemetryString
        }

        switch stateCopy.attestationsReceivedState {
        case .noAttestationReceived:
            break
        case .receivedAttestations(count: _, let durationSinceStart):
            requestMetrics.fields[.attestationFetchTime] = .milliSeconds(durationSinceStart)
            requestMetrics.fields[.attestationIsFirstFetch] = true
        }

        switch stateCopy.ropesRequestSentState {
        case .notSend:
            break
        case .succeeded(let durationSinceStart):
            requestMetrics.fields[.invokeRequestSendTime] = .milliSeconds(durationSinceStart)
            requestMetrics.fields[.invokeRequestSendSuccess] = true
        case .failed(let error, let durationSinceStart):
            requestMetrics.fields[.invokeRequestSendTime] = .milliSeconds(durationSinceStart)
            requestMetrics.fields[.invokeRequestSendSuccess] = false
            requestMetrics.fields[.invokeRequestSendError] = error.telemetryString
        }

        if let durationSinceStartTillNodeSelected = stateCopy.durationSinceStartTillNodeSelected {
            requestMetrics.fields[.nodeSelectedTime] = .milliSeconds(durationSinceStartTillNodeSelected)
        }

        let cachedAttestationCount = stateCopy.nodes.values.reduce(into: 0, { $0 += $1.isFromCache ? 1 : 0 })
        requestMetrics.fields[.hasCachedAttestations] = .bool(cachedAttestationCount > 0)
        requestMetrics.fields[.cachedAttestationCount] = .int(Int64(cachedAttestationCount))

        let verifiedAttestationCount = stateCopy.nodes.values.reduce(
            into: 0,
            { result, element in
                switch element.state {
                case .unverified, .verifying, .verifiedFailed:
                    break
                case .finished, .receiving, .sentKey, .verified:
                    result += 1
                }
            })

        requestMetrics.fields[.verifiedAttestationCount] = .int(Int64(verifiedAttestationCount))

        switch stateCopy.firstChunkSentState {
        case .notSend:
            break
        case .succeeded(let duration, let withinBudget):
            requestMetrics.fields[.firstChunkSendWithinBudget] = .bool(withinBudget)
            requestMetrics.fields[.firstChunkSendTime] = .milliSeconds(duration)
        }

        if let durationSinceStartTillLastPayloadChunkSent = stateCopy.durationSinceStartTillLastPayloadChunkSent {
            requestMetrics.fields[.remainingChunkSendTime] = .milliSeconds(durationSinceStartTillLastPayloadChunkSent)
        }

        if let durationSinceStartTillFirstToken = stateCopy.durationSinceStartTillFirstToken {
            requestMetrics.fields[.firstResponseReceivedTime] = .milliSeconds(durationSinceStartTillFirstToken)
        }

        switch stateCopy.ropesRequestState {
        case .initialized, .waitingForConnection, .requestSent, .responseHeadReceived, .attestationsReceived, .nodeSelected:
            break
        case .finished(let duration):
            requestMetrics.fields[.trustedRequestTotalTime] = .milliSeconds(duration)
            requestMetrics.fields[.trustedRequestSuccess] = true
        case .failed(let error, let duration):
            requestMetrics.fields[.trustedRequestTotalTime] = .milliSeconds(duration)
            requestMetrics.fields[.trustedRequestError] = error.telemetryString
            requestMetrics.fields[.trustedRequestSuccess] = false
        }

        switch stateCopy.kDataSendState {
        case .notSend:
            break
        case .sent(let duration, let count):
            requestMetrics.fields[.kDataSendTime] = .milliSeconds(duration)
            requestMetrics.fields[.kDataSendCount] = .int(Int64(count))
        }

        return requestMetrics
    }

    // MARK: - Connection metrics

    func reportConnectionReady() {
        let duration = self.startInstant.duration(to: self.clock.now)

        self.state.withLock { state in
            switch state.connectionEstablishState {
            case .notEstablished, .failed:
                state.connectionEstablishState = .established(duration, nil)

            case .established:
                break
            }
        }
    }

    func reportConnectionEstablishReport(_ report: NWConnection.EstablishmentReport) {
        let duration = self.startInstant.duration(to: self.clock.now)

        self.state.withLock { state in
            switch state.connectionEstablishState {
            case .notEstablished, .failed:
                // this should never happen, but we don't want to crash here!
                state.connectionEstablishState = .established(duration, report)

            case .established(_, .some):
                // this should never happen, but we don't want to crash here!
                break

            case .established(let duration, .none):
                state.connectionEstablishState = .established(duration, report)
            }
        }
    }

    func reportConnectionError(_ error: any Error) {
        let duration = self.startInstant.duration(to: self.clock.now)

        self.state.withLock { state in
            switch state.connectionEstablishState {
            case .notEstablished:
                state.connectionEstablishState = .failed(duration, error)

            case .established, .failed:
                // let's not overwrite the initial state!
                break
            }
        }
    }

    // MARK: - Ropes Connection

    func attachNetworkActivities(_ activityStarter: some NWActivityTracker) {
        let attestationActivity = NWActivity(domain: .cloudCompute, label: .attestationFetch)
        let firstTokenActivity = NWActivity(domain: .cloudCompute, label: .computeRequest)

        activityStarter.startActivity(attestationActivity)
        activityStarter.startActivity(firstTokenActivity)

        self.state.withLock { state in
            switch state.responseStreamState {
            case .initialized:
                state.responseStreamState = .waitingToSendFirstKey(firstTokenActivity: firstTokenActivity)
            case .waitingToSendFirstKey, .waitingForNode, .receiving, .finished, .failed:
                // TODO: fail the activity right away
                break
            }

            switch state.ropesRequestState {
            case .initialized:
                state.ropesRequestState = .waitingForConnection(attestationsActivity: attestationActivity)

            case .waitingForConnection,
                .requestSent,
                .responseHeadReceived,
                .attestationsReceived,
                .nodeSelected,
                .finished,
                .failed:
                // TODO: fail the activity right away
                break
            }
        }
    }

    func observeSendingRopesRequest<Success: Sendable>(_ closure: () async throws -> Success) async throws -> Success {
        let result = await Result(asyncCatching: closure)
        let duration = self.startInstant.duration(to: self.clock.now)

        self.state.withLock {
            switch result {
            case .success:
                $0.ropesRequestSentState = .succeeded(duration)
                guard case .waitingForConnection(let activity) = $0.ropesRequestState else {
                    break
                }
                $0.ropesRequestState = .requestSent(attestationsActivity: activity)
            case .failure(let error):
                $0.ropesRequestSentState = .failed(error, duration)
            // NOTE: ropesRequestState failure will be set through surrounding metrics calls
            }
        }
        if result.isSuccess {
            self.signposter.emitEvent("RopesInvokeRequestSent", id: self.signpostID)
        }
        self.logger.log("\(self.lp) Ropes invoke request sent")
        return try result.get()
    }

    func ropesConnectionResponseReceived(_ httpResponse: HTTPResponse) {
        let isResponseHead = self.state.withLock {
            guard case .requestSent(let attestationsActivity) = $0.ropesRequestState else {
                return false
            }

            $0.ropesRequestState = .responseHeadReceived(attestationsActivity: attestationsActivity)
            $0.responseCode = httpResponse.status.code
            $0.ropesVersion = httpResponse.headerFields[.appleServerBuildVersion]
            return true
        }

        guard isResponseHead else { return }

        self.logger.log("\(self.lp) Ropes invoke response head received")
        self.signposter.emitEvent("RopesResponseHeadReceived", id: self.signpostID)

        var invokeResponseMetric = TC2InvokeResponseMetric(bundleID: self.bundleID)
        invokeResponseMetric.fields[.clientRequestid] = .string(self.requestIDForEventReporting.uuidString)
        invokeResponseMetric.fields[.eventTime] = .int(Int64(Date().timeIntervalSince1970))
        invokeResponseMetric.fields[.environment] = .string(self.environment)
        invokeResponseMetric.fields[.clientInfo] = .string(self.clientInfo)
        if let featureID = self.featureID {
            invokeResponseMetric.fields[.featureID] = .string(featureID)
        }
        invokeResponseMetric.fields[.bundleID] = .string(self.bundleID)
        if let originatingBundleID = self.originatingBundleID {
            invokeResponseMetric.fields[.originatingBundleID] = .string(originatingBundleID)
        }
        invokeResponseMetric.fields[.locale] = .string(Locale.current.identifier)
        invokeResponseMetric.fields[.invokeResponseSuccess] = true
        self.eventStreamContinuation.yield(.exportMetric(invokeResponseMetric))
    }

    func requestFinished(error maybeError: TrustedCloudComputeError?) async {
        let duration = self.startInstant.duration(to: self.clock.now)
        self.state.withLock { state in
            switch maybeError {
            case nil:
                state.ropesRequestState = .finished(duration)

            case let error?:
                state.ropesRequestState = .failed(error, duration)
            }
        }

        self.signposter.endInterval("FullTrustedRequest", self.fullRequestInterval)

        let requestEvent = self.makeFullRequestMetrics()
        self.eventStreamContinuation.yield(.exportMetric(requestEvent))
        await self.logOSLogAndBiomeStreamRequestLog()
    }

    func observeAuthTokenFetch<Success: Sendable>(_ closure: () async throws -> Success) async throws -> Success {
        let result = await self.signposter.withIntervalSignpost("FetchOTT", id: self.signpostID) {
            await Result(asyncCatching: closure)
        }

        let duration = self.startInstant.duration(to: self.clock.now)
        // need to get a copy here, as we don't want to enforce that Success is Sendable.
        // `withLock` enforces a Sendable closure for whatever reason.
        self.state.withLock {
            switch result {
            case .success:
                $0.authTokenFetchState = .succeeded(duration)
            case .failure(let error):
                // If there is a failure to fetch the tokens, we want to know why for
                // this telemetry. The outer error will show in the trusted request failure.
                if let trustedRequestError = error as? TrustedRequestError {
                    $0.authTokenFetchState = .failed(trustedRequestError.selfOrFirstUnderlying, duration)
                } else {
                    $0.authTokenFetchState = .failed(error, duration)
                }
            }
        }
        return try result.get()
    }

    // MARK: - Data streams calls

    func observeAuthTokenSend<Success: Sendable>(_ closure: () async throws -> Success) async throws -> Success {
        let result = await Result(asyncCatching: closure)
        let durationSinceStartTillTokenGrantingTokenSent = self.startInstant.duration(to: self.clock.now)
        self.state.withLock {
            switch result {
            case .success:
                $0.authTokenSendState = .succeeded(durationSinceStartTillTokenGrantingTokenSent)
                $0.dataStreamState = .authTokenSent
            case .failure(let error):
                $0.authTokenSendState = .failed(error, durationSinceStartTillTokenGrantingTokenSent)
                $0.dataStreamState = .failed(error)
            }
        }

        if result.isSuccess {
            self.signposter.emitEvent("OTTSent", id: self.signpostID)
            self.logger.log("\(self.lp) Sent auth message on data stream")
        }

        return try result.get()
    }

    func nodeSelected() {
        let duration = self.startInstant.duration(to: self.clock.now)
        self.state.withLock {
            switch $0.dataStreamState {
            case .initialized, .authTokenSent:
                $0.dataStreamState = .nodeSelected(bytesSent: 0)
                $0.durationSinceStartTillNodeSelected = duration
            case .awaitingNodeSelected(let bytesSent):
                $0.dataStreamState = .nodeSelected(bytesSent: bytesSent)
                $0.durationSinceStartTillNodeSelected = duration
            case .finished:
                // it can happen that we get a nodeSelected _after_ the data stream has finished.
                // this happens in scenarios where we are able to send the complete payload within
                // the initial budget
                $0.durationSinceStartTillNodeSelected = duration

            case .nodeSelected, .failed:
                break
            }
        }
        self.signposter.emitEvent("NodeSelected", id: self.signpostID)
        self.logger.log("\(self.lp): nodeSelected received")
    }

    func receivedOutgoingUserDataChunk() {
        self.signposter.emitEvent("ReceivedOutgoingUserDataChunk", id: self.signpostID)
    }

    func observeDataWrite<Success: Sendable>(
        bytesToSend: Int,
        inBudget: Bool = true,
        _ closure: () async throws -> Success
    ) async throws -> Success {
        let result = await Result(asyncCatching: closure)
        let durationSinceStartTillNow = self.startInstant.duration(to: self.clock.now)
        self.state.withLock {
            var isFirstWrite: Bool = false
            switch result {
            case .success:
                switch $0.dataStreamState {
                case .initialized:
                    fatalError("Invalid state: \($0.dataStreamState). Auth token must be sent first!")
                case .authTokenSent:
                    $0.dataStreamState = .awaitingNodeSelected(bytesSent: bytesToSend)
                    isFirstWrite = true

                case .awaitingNodeSelected(let bytesSent):
                    $0.dataStreamState = .awaitingNodeSelected(bytesSent: bytesSent + bytesToSend)

                case .nodeSelected(let bytesSent):
                    $0.dataStreamState = .nodeSelected(bytesSent: bytesSent + bytesToSend)
                    isFirstWrite = bytesSent == 0

                case .finished, .failed:
                    break  // invalid call, but we don't want to crash here!
                }

                if isFirstWrite {
                    $0.firstChunkSentState = .succeeded(durationSinceStartTillNow, withinBudget: inBudget)
                }

            case .failure(let failure):
                switch $0.dataStreamState {
                case .initialized, .authTokenSent, .awaitingNodeSelected, .nodeSelected:
                    $0.dataStreamState = .failed(failure)
                case .finished, .failed:
                    break
                }
            }
        }

        return try result.get()
    }

    func dataStreamFinished() {
        let durationSinceStartTillNow = self.startInstant.duration(to: self.clock.now)
        self.state.withLock {
            switch $0.dataStreamState {
            case .initialized:
                fatalError("Invalid state: \($0.dataStreamState). Auth token must be sent first!")
            case .authTokenSent:
                $0.dataStreamState = .finished(bytesSent: 0)
                $0.durationSinceStartTillLastPayloadChunkSent = durationSinceStartTillNow

            case .awaitingNodeSelected(let bytesSent):
                // it can happen that we finish the datastream _before_ we got a nodeSelected.
                // this happens in scenarios where we are able to send the complete payload within
                // the initial budget
                $0.dataStreamState = .finished(bytesSent: bytesSent)
                $0.durationSinceStartTillLastPayloadChunkSent = durationSinceStartTillNow

            case .nodeSelected(let bytesSent):
                $0.dataStreamState = .finished(bytesSent: bytesSent)
                $0.durationSinceStartTillLastPayloadChunkSent = durationSinceStartTillNow

            case .finished, .failed:
                break  // invalid call, but we don't want to crash here!
            }
        }
    }

    // MARK: - Node stream calls

    func observeLoadAttestationsFromCache(closure: () async -> [ValidatedAttestationOrAttestation]) async -> [ValidatedAttestationOrAttestation] {
        let result = await self.signposter.withIntervalSignpost("LoadAttestationsFromCache", id: self.signpostID) {
            await Result(asyncCatching: closure)
        }

        self.state.withLock { state in
            switch result {
            case .success(let attestations):
                for attestation in attestations {
                    switch attestation {
                    case .inlineAttestation(let attestation, let ohttpContext):
                        if let bundle = attestation.attestationBundle {
                            state.nodes[attestation.nodeID] = .init(
                                state: .unverified,
                                nodeID: attestation.nodeID,
                                attestationBundleRef: .data(bundle),
                                ohttpContext: UInt64(ohttpContext),
                                cloudOSVersion: attestation.cloudOSVersion,
                                cloudOSReleaseType: attestation.cloudOSReleaseType,
                                maybeValidatedCellID: attestation.unvalidatedCellID,
                                ensembleID: attestation.ensembleID
                            )
                        } else {
                            self.logger.error("bundle missing for attestation: \(attestation.nodeID)")
                        }
                    case .cachedValidatedAttestation(let validatedAttestation, let ohttpContext):
                        state.nodes[validatedAttestation.attestation.nodeID] = .init(
                            state: .verified,
                            nodeID: validatedAttestation.attestation.nodeID,
                            attestationBundleRef: .lookupInDatabase,
                            ohttpContext: UInt64(ohttpContext),
                            cloudOSVersion: validatedAttestation.attestation.cloudOSVersion,
                            cloudOSReleaseType: validatedAttestation.attestation.cloudOSReleaseType,
                            maybeValidatedCellID: validatedAttestation.validatedCellID,
                            ensembleID: validatedAttestation.attestation.ensembleID
                        )
                    }
                }
            }
        }

        return result.get()
    }

    func attestationsReceived(_ attestations: [ValidatedAttestationOrAttestation]) {
        let duration = self.startInstant.duration(to: self.clock.now)
        let count = attestations.count
        let maybeAttestationActivity = self.state.withLock { state -> NWActivity? in
            var result: NWActivity? = nil
            switch state.attestationsReceivedState {
            case .noAttestationReceived:
                state.attestationsReceivedState = .receivedAttestations(count: count, durationSinceStart: duration)
            case .receivedAttestations(let existing, let durationSinceStart):
                state.attestationsReceivedState = .receivedAttestations(count: existing + count, durationSinceStart: durationSinceStart)
            }

            switch state.ropesRequestState {
            case .responseHeadReceived(let attestationsActivity):
                // expected case
                state.ropesRequestState = .attestationsReceived
                result = attestationsActivity

            case .attestationsReceived:
                // expected case. nothing to do. got a second attestation message
                break

            case .waitingForConnection(let attestationsActivity), .requestSent(let attestationsActivity):
                // unexpected case. but we should not crash here!
                state.ropesRequestState = .attestationsReceived
                result = attestationsActivity

            case .initialized, .nodeSelected, .finished, .failed:
                // unexpected case. nothing to do
                break
            }

            for attestation in attestations {
                switch attestation {
                case .cachedValidatedAttestation:
                    // We will never receive validated attestations
                    self.logger.error("Received unexpected validated attestation nodeID: \(attestation.identifier)")
                    break

                case .inlineAttestation(let attestation, let ohttpContext):
                    guard state.nodes[attestation.nodeID] == nil else { continue }
                    if let bundle = attestation.attestationBundle {
                        state.nodes[attestation.nodeID] = .init(
                            state: .unverified,
                            nodeID: attestation.nodeID,
                            attestationBundleRef: .data(bundle),
                            ohttpContext: UInt64(ohttpContext),
                            cloudOSVersion: attestation.cloudOSVersion,
                            cloudOSReleaseType: attestation.cloudOSReleaseType,
                            maybeValidatedCellID: attestation.unvalidatedCellID,
                            ensembleID: attestation.ensembleID
                        )
                    } else {
                        self.logger.error("bundle missing for attestation: \(attestation.nodeID)")
                    }
                }
            }
            return result
        }
        self.signposter.emitEvent("AttestationsReceivedFromRopes", id: self.signpostID)
        maybeAttestationActivity?.complete(reason: .success)
    }

    func inlineAttestationsValidated(_ nodeIDs: [String]) {
        self.eventStreamContinuation.yield(.nodesReceived(nodeIDs: nodeIDs, fromSource: .request))
    }

    func noFurtherAttestations() {
        let maybeAttestationActivity = self.state.withLock { state -> NWActivity? in
            switch state.ropesRequestState {
            case .responseHeadReceived(let attestationsActivity):
                // expected case. got no attestations from ROPES. Cache was sufficient.
                state.ropesRequestState = .attestationsReceived
                return attestationsActivity

            case .attestationsReceived:
                // expected case. got attestations before. Cache was insufficient.
                return nil

            case .waitingForConnection(let attestationsActivity), .requestSent(let attestationsActivity):
                // unexpected case. but we should not crash here!
                state.ropesRequestState = .attestationsReceived
                return attestationsActivity

            case .initialized, .nodeSelected, .finished, .failed:
                // unexpected case. nothing to do
                return nil
            }
        }
        maybeAttestationActivity?.complete(reason: .success)
    }

    func observeAttestationVerify(nodeID: String, closure: () async throws -> ValidatedAttestation) async throws -> ValidatedAttestation {
        self.state.withLock { state in
            guard var value = state.nodes[nodeID] else { return }
            value.state = .verifying
            state.nodes[nodeID] = value
        }

        // We are verifying attestations in parallel, so we need to create one signpostID for each verification.
        let attestationSignpostID = self.signposter.makeSignpostID()
        let startTime = self.clock.now
        let result = await self.signposter.withIntervalSignpost("VerifyAttestation", id: attestationSignpostID) {
            await Result(asyncCatching: closure)
        }
        let duration = startTime.duration(to: self.clock.now)

        self.state.withLock { state in
            guard var value = state.nodes[nodeID] else { return }
            guard case .verifying = value.state else { return }

            switch result {
            case .success:
                value.state = .verified
                state.verifiedAttestationsCount += 1

            case .failure(let error):
                value.state = .verifiedFailed(error)
            }

            state.nodes[nodeID] = value
        }

        var verificationMetric = TC2AttestationVerificationMetric(bundleID: self.bundleID)
        verificationMetric.fields[.clientRequestid] = .string(self.requestIDForEventReporting.uuidString)
        verificationMetric.fields[.eventTime] = .int(Int64(Date().timeIntervalSince1970))
        verificationMetric.fields[.environment] = .string(self.environment)
        verificationMetric.fields[.clientInfo] = .string(self.clientInfo)
        if let featureID = self.featureID {
            verificationMetric.fields[.featureID] = .string(featureID)
        }
        verificationMetric.fields[.bundleID] = .string(self.bundleID)
        if let originatingBundleID = self.originatingBundleID {
            verificationMetric.fields[.originatingBundleID] = .string(originatingBundleID)
        }
        verificationMetric.fields[.locale] = .string(self.locale)
        verificationMetric.fields[.isPrefetchedAttestation] = false
        verificationMetric.fields[.attestationVerificationTime] = .milliSeconds(duration)
        switch result {
        case .success:
            verificationMetric.fields[.attestationVerificationSuccess] = true
        case .failure(let error):
            verificationMetric.fields[.attestationVerificationNodeIdentifier] = .string(nodeID)
            verificationMetric.fields[.attestationVerificationSuccess] = false
            verificationMetric.fields[.attestationVerificationError] = error.telemetryString

            // we also need to report a separate event for the error
            var verifivationErrorMetric = TC2AttestationnVerificationErrorMetric(bundleID: self.bundleID)
            verifivationErrorMetric.fields[.clientRequestid] = .string(self.requestIDForEventReporting.uuidString)
            verifivationErrorMetric.fields[.eventTime] = .int(Int64(Date().timeIntervalSince1970))
            verifivationErrorMetric.fields[.environment] = .string(self.environment)
            verifivationErrorMetric.fields[.clientInfo] = .string(self.clientInfo)
            if let featureID = self.featureID {
                verifivationErrorMetric.fields[.featureID] = .string(featureID)
            }
            verifivationErrorMetric.fields[.bundleID] = .string(self.bundleID)
            if let originatingBundleID = self.originatingBundleID {
                verifivationErrorMetric.fields[.originatingBundleID] = .string(originatingBundleID)
            }
            verifivationErrorMetric.fields[.locale] = .string(self.locale)
            // this is false because we are on request flow here
            verifivationErrorMetric.fields[.isPrefetchedAttestation] = false
            verifivationErrorMetric.fields[.attestationVerificationNodeIdentifier] = .string(nodeID)
            verifivationErrorMetric.fields[.attestationVerificationError] = error.telemetryString
            verifivationErrorMetric.fields[.attestationVerificationTime] = .milliSeconds(duration)
            self.eventStreamContinuation.yield(.exportMetric(verifivationErrorMetric))
        }
        self.eventStreamContinuation.yield(.exportMetric(verificationMetric))

        return try result.get()
    }

    func observeSendingKeyToNode(nodeID: String, _ closure: () async throws -> Void) async throws {
        let result = await Result(asyncCatching: closure)

        var kdataSendMetrics = TC2KDataSendMetric()
        kdataSendMetrics.fields[.clientRequestid] = .string(self.requestIDForEventReporting.uuidString)
        kdataSendMetrics.fields[.eventTime] = .int(Int64(Date().timeIntervalSince1970))
        kdataSendMetrics.fields[.environment] = .string(self.environment)
        kdataSendMetrics.fields[.clientInfo] = .string(self.clientInfo)
        if let featureID = self.featureID {
            kdataSendMetrics.fields[.featureID] = .string(featureID)
        }
        kdataSendMetrics.fields[.bundleID] = .string(self.bundleID)
        if let originatingBundleID = self.originatingBundleID {
            kdataSendMetrics.fields[.originatingBundleID] = .string(originatingBundleID)
        }
        kdataSendMetrics.fields[.locale] = .string(self.locale)
        kdataSendMetrics.fields[.kdataSendSuccess] = .bool(result.isSuccess)
        if let error = result.maybeError {
            kdataSendMetrics.fields[.kdataSendNodeIdentifier] = .string(nodeID)
            kdataSendMetrics.fields[.kdataSendError] = error.telemetryString
        }
        self.eventStreamContinuation.yield(.exportMetric(kdataSendMetrics))

        guard result.isSuccess else { return }
        let duration = self.startInstant.duration(to: self.clock.now)
        let firstKeySent = self.state.withLock { state -> Bool in
            guard var value = state.nodes[nodeID] else { return false }
            guard case .verified = value.state else { return false }
            value.state = .sentKey
            state.nodes[nodeID] = value

            // update kDataSendState to record how many keys are we sending to and when did we send the first key
            var newKDataSendState = state.kDataSendState
            switch state.kDataSendState {
            case .notSend:
                newKDataSendState = .sent(duration: duration, count: 1)
            case .sent(let firstSentDuration, let count):
                // just increase the counter
                newKDataSendState = .sent(duration: firstSentDuration, count: count + 1)
            }
            state.kDataSendState = newKDataSendState

            switch state.responseStreamState {
            case .initialized:
                // invalid state. don't crash though
                return false

            case .waitingToSendFirstKey(let firstTokenActivity):
                let interval = self.signposter.beginInterval("SentKey", id: self.signpostID)
                state.responseStreamState = .waitingForNode(firstTokenActivity: firstTokenActivity, interval: interval)
                return true

            case .waitingForNode, .receiving, .finished, .failed:
                // we sent the key to another node first
                return false
            }
        }

        self.signposter.emitEvent("SentKeyToNode", id: self.signpostID)
        if firstKeySent {
            self.logger.log("\(self.lp) First key sent to node.")
        }
    }

    private enum NodeResponseStreamFirstMessageAction: Sendable {
        case endActivity(NWActivity)
        case endActivityAndEndInterval(NWActivity, OSSignpostIntervalState, intervalName: StaticString)
    }

    func nodeFirstResponseReceived(nodeID: String) {
        let action = self.state.withLock { state -> NodeResponseStreamFirstMessageAction? in
            guard var value = state.nodes[nodeID] else { return nil }
            guard case .sentKey = value.state else { return nil }
            value.state = .receiving(summaryReceived: false, bytesReceived: 0)
            state.nodes[nodeID] = value

            switch state.responseStreamState {
            case .waitingForNode(let firstTokenActivity, let interval):
                let receivingInterval = self.signposter.beginInterval("NodeResponse", id: self.signpostID)
                state.responseStreamState = .receiving(nodeID: nodeID, bytesReceived: 0, interval: receivingInterval)
                return .endActivityAndEndInterval(firstTokenActivity, interval, intervalName: "SentKey")

            case .waitingToSendFirstKey(let firstTokenActivity):
                // unexpected! but we don't want to crash here!
                let receivingInterval = self.signposter.beginInterval("NodeResponse", id: self.signpostID)
                state.responseStreamState = .receiving(nodeID: nodeID, bytesReceived: 0, interval: receivingInterval)
                return .endActivity(firstTokenActivity)

            case .receiving:
                return nil

            case .initialized, .finished, .failed:
                return nil
            }
        }

        switch action {
        case .none:
            break
        case .endActivity(let activity):
            activity.complete(reason: .success)
        case .endActivityAndEndInterval(let activity, let interval, let intervalName):
            activity.complete(reason: .success)
            self.signposter.endInterval(intervalName, interval)
        }

        var endpointResponseMetric = TC2TrustedEndpointResponseMetric(bundleID: self.bundleID)
        endpointResponseMetric.fields[.clientRequestid] = .string(self.requestIDForEventReporting.uuidString)
        endpointResponseMetric.fields[.eventTime] = .int(Int64(Date().timeIntervalSince1970))
        endpointResponseMetric.fields[.environment] = .string(self.environment)
        endpointResponseMetric.fields[.clientInfo] = .string(self.clientInfo)
        if let featureID = self.featureID {
            endpointResponseMetric.fields[.featureID] = .string(featureID)
        }
        endpointResponseMetric.fields[.bundleID] = .string(self.bundleID)
        if let originatingBundleID = self.originatingBundleID {
            endpointResponseMetric.fields[.originatingBundleID] = .string(originatingBundleID)
        }
        endpointResponseMetric.fields[.locale] = .string(Locale.current.identifier)
        endpointResponseMetric.fields[.trustedEndpointResponseSuccess] = true
        endpointResponseMetric.fields[.trustedEndpointResponseNodeIdentifier] = .string(nodeID)
        self.eventStreamContinuation.yield(.exportMetric(endpointResponseMetric))
    }

    func nodeSummaryReceived(nodeID: String) {
        self.state.withLock { state in
            guard var value = state.nodes[nodeID] else { return }
            guard case .receiving(false, let bytesReceived) = value.state else { return }
            value.state = .receiving(summaryReceived: true, bytesReceived: bytesReceived)
            state.nodes[nodeID] = value
        }
    }

    func nodeResponsePayloadReceived(nodeID: String, bytes: Int) {
        self.state.withLock { state in
            guard var value = state.nodes[nodeID] else { return }
            guard case .receiving(let summaryReceived, let bytesReceived) = value.state else { return }
            value.state = .receiving(summaryReceived: summaryReceived, bytesReceived: bytesReceived + bytes)
            state.nodes[nodeID] = value

            switch state.responseStreamState {
            case .receiving(let nodeID, let bytesReceived, let interval):
                if bytesReceived == 0 {  // we have now received our first token
                    let duration = self.startInstant.duration(to: self.clock.now)
                    state.durationSinceStartTillFirstToken = duration
                }
                state.responseStreamState = .receiving(nodeID: nodeID, bytesReceived: bytesReceived + bytes, interval: interval)

            case .initialized, .finished, .failed, .waitingForNode, .waitingToSendFirstKey:
                // invalid states! Don't crash though!
                break
            }
        }
    }

    func nodeResponseFinished(nodeID: String) {
        let receiveInterval = self.state.withLock { state -> OSSignpostIntervalState? in
            guard var value = state.nodes[nodeID] else { return nil }
            guard case .receiving(let summaryReceived, let bytesReceived) = value.state else { return nil }
            value.state = .finished(summaryReceived: summaryReceived, bytesReceived: bytesReceived)
            state.nodes[nodeID] = value

            switch state.responseStreamState {
            case .initialized, .waitingToSendFirstKey, .waitingForNode, .finished, .failed:
                // invalid states! Don't crash though!
                return nil

            case .receiving(let nodeID, let bytesReceived, let interval):
                state.responseStreamState = .finished(nodeID: nodeID, bytesReceived: bytesReceived)
                return interval
            }
        }

        if let receiveInterval {
            self.signposter.endInterval("NodeResponse", receiveInterval)
        }
    }

    private enum NodeResponseStreamFailedAction: Sendable {
        case failActivity(NWActivity)
        case failActivityAndEndInterval(NWActivity, OSSignpostIntervalState, intervalName: StaticString)
        case endInterval(OSSignpostIntervalState, intervalName: StaticString)
    }

    func nodeResponseStreamsFailed(_ error: any Error) {
        let action = self.state.withLock { state -> NodeResponseStreamFailedAction? in
            // If there is a TrustedRequestError in attestation validation, we want to know
            // what happened in this telemetry. In that case, the outer error will show
            // the trusted request failure.
            let trustedRequestError = error as? TrustedRequestError
            let error = trustedRequestError?.selfOrFirstUnderlying ?? error

            switch state.responseStreamState {
            case .initialized:
                state.responseStreamState = .failed(error)
                return nil
            case .waitingToSendFirstKey(let firstTokenActivity):
                state.responseStreamState = .failed(error)
                return .failActivity(firstTokenActivity)
            case .waitingForNode(let firstTokenActivity, let interval):
                state.responseStreamState = .failed(error)
                return .failActivityAndEndInterval(firstTokenActivity, interval, intervalName: "SentKey")
            case .receiving(_, _, let interval):
                state.responseStreamState = .failed(error)
                return .endInterval(interval, intervalName: "ReceivingResponse")
            case .finished, .failed:
                return nil
            }
        }

        switch action {
        case .none:
            break
        case .endInterval(let interval, let intervalName):
            self.signposter.endInterval(intervalName, interval)
        case .failActivity(let activity):
            activity.complete(reason: .failure)
        case .failActivityAndEndInterval(let activity, let interval, let intervalName):
            activity.complete(reason: .failure)
            self.signposter.endInterval(intervalName, interval)
        }
    }

    // MARK: - Private Method -

    private func logOSLogAndBiomeStreamRequestLog() async {
        // log the metadata associated with this request for researcher logs:
        // 1. all the request parameters {model, arguments}
        // 2. a deserialization of all attestations, including failed nodes
        //    this is provided by CloudAttestation

        if !TransparencyReport().enabled {
            self.logger.log("\(self.lp): Request Log: TransparencyReport is not enabled")
            return
        }

        self.logger.log("\(self.lp): Request Log: logging request data for analysis")
        self.logger.log("\(self.lp): Request Log: request parameters:")
        self.logger.log("\(self.lp): Request Log: pipelineKind: \(self.parameters.pipelineKind)")
        self.logger.log("\(self.lp): Request Log: pipelineArguments: \(self.parameters.pipelineArguments)")
        self.logger.log("\(self.lp): Request Log: attestations: ")
        let nodes = self.state.withLock { state in state.nodes }

        // Load bundles from store
        var bundles: [String: Data] = [:]
        if let store = self.attestationStore {
            bundles = await store.getAttestationBundlesUsedByTrustedRequest(serverRequestID: self.serverRequestID)
        }

        var biomeAttestations: [PrivateCloudComputeRequestLog.Attestation] = []
        biomeAttestations.reserveCapacity(nodes.count)
        for (key, node) in nodes {
            var attestationString: String = ""
            switch node.attestationBundleRef {
            case .lookupInDatabase:
                if let bundle = bundles[node.nodeID] {
                    attestationString = self.makeAttestationString(attestationBundle: bundle)
                }
            case .data(let bundle):
                attestationString = self.makeAttestationString(attestationBundle: bundle)
            }

            let validatedString =
                switch node.state {
                case .unverified, .verifying, .verifiedFailed:
                    "Unvalidated"
                case .verified,
                    .sentKey,
                    .receiving,
                    .finished:
                    "Validated"
                }
            self.logger.log("\(self.lp): Request Log: Attestation: \(key) \(node.state) <\(validatedString) \(node.nodeID): \(attestationString)>")

            var biomeAttestation: PrivateCloudComputeRequestLog.Attestation = .init()
            biomeAttestation.node = node.nodeID
            biomeAttestation.nodeState = validatedString
            biomeAttestation.attestationString = attestationString
            biomeAttestations.append(biomeAttestation)
        }

        // Log the request parameters to Biome stream
        do {
            let workloadParametersAsJSON = try self.encoder.encode(self.parameters.pipelineArguments)
            let workloadParametersAsString = String(data: workloadParametersAsJSON, encoding: .utf8) ?? ""
            let stream = Library.Streams.PrivateCloudCompute.RequestLog.self
            #if os(macOS)
            let userID = geteuid()
            logger.debug("Biome stream source created with user=\(Int(userID))")
            let source = try stream.source(user: userID)
            #else
            let source = try stream.source()
            #endif
            // The Biome stream gets the clientRequestID.
            source.sendEvent(
                .with {
                    $0.requestId = self.clientRequestID.uuidString
                    $0.timestamp = Date()
                    $0.pipelineKind = self.parameters.pipelineKind
                    $0.pipelineParameters = workloadParametersAsString
                    $0.attestations = biomeAttestations
                })
        } catch {
            self.logger.error("failed to log Biome event: \(error)")
        }
    }

    private func makeAttestationString(attestationBundle: Data) -> String {
        // We should move this out of the request and into a helper method but for now
        // this is better than in the AttestationVerifier
        var bundle: AttestationBundle
        var attestationString: String = ""
        do {
            do {
                bundle = try AttestationBundle(data: attestationBundle)
                do {
                    attestationString = try bundle.jsonString()
                } catch {
                    self.logger.error("bundle.jsonString failed: \(error)")
                }
            } catch {
                self.logger.error("AttestationBundle.init failed: \(error)")
            }
        }
        return attestationString
    }
}

extension Result {
    fileprivate var isSuccess: Bool {
        switch self {
        case .success: true
        case .failure: false
        }
    }

    fileprivate var maybeError: Failure? {
        switch self {
        case .success: nil
        case .failure(let failure): failure
        }
    }
}

extension EventValue {
    /// return milliseconds value of the duration in integet event value
    static func milliSeconds(_ duration: Duration) -> EventValue {
        let seconds = duration.components.seconds * 1000
        let milliseconds = Int64(duration.components.attoseconds / 1_000_000_000_000_000)
        return .int(seconds + milliseconds)
    }
}

extension OSSignposter {
    func withIntervalSignpost<T>(
        _ name: StaticString,
        id: OSSignpostID = .exclusive,
        around task: () async throws -> T
    ) async rethrows -> T {
        let interval = self.beginInterval(name, id: id)
        defer { self.endInterval(name, interval) }
        return try await task()
    }
}

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
@_spi(SEP_Curve25519) import CryptoKit
@_spi(SEP_Curve25519) import CryptoKitPrivate
import Foundation
import LocalAuthentication
import ObliviousX
import os

/// Messenger responsible for communication with cloudboardd
actor CloudBoardMessenger: CloudBoardJobHelperAPIClientToServerProtocol, CloudBoardJobHelperAPIServerDelegateProtocol,
CloudBoardAttestationAPIClientDelegateProtocol {
    public static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "CloudBoardMessenger"
    )

    enum CloudBoardMessengerError: Error {
        case runNeverCalled
    }

    /// The metrics system to use.
    private let metrics: MetricsSystem
    private let laContext: LAContext

    /// Captures state of retrieving the OHTTP node keys from the attestation daemon allowing the messenger to wait
    /// for the key to be retrieved from the attestation daemon when it receives workload requests before the keys are
    /// available
    private enum OHTTPKeyState {
        case initialized
        case available([CachedAttestedKey])
        case awaitingKeys(Promise<[CachedAttestedKey], Error>)
    }

    private var ohttpKeyState = OHTTPKeyState.initialized
    var ohttpKeys: [CachedAttestedKey] {
        get async throws {
            switch self.ohttpKeyState {
            case .initialized:
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Waiting for OHTTP keys to become available"
                ).log(to: Self.logger, level: .default)
                let promise = Promise<[CachedAttestedKey], Error>()
                self.ohttpKeyState = .awaitingKeys(promise)
                return try await Future(promise).valueWithCancellation
            case .available(let keys):
                return keys
            case .awaitingKeys(let promise):
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Waiting for OHTTP keys to become available"
                ).log(to: Self.logger, level: .default)
                return try await Future(promise).valueWithCancellation
            }
        }
    }

    let attestationClient: CloudBoardAttestationAPIClientProtocol?

    let server: CloudBoardJobHelperAPIServerToClientProtocol
    let encodedRequestContinuation: AsyncStream<PipelinePayload<Data>>.Continuation
    let responseStream: AsyncStream<FinalizableChunk<Data>>

    var responseEncapsulator: OHTTPEncapsulation.StreamingResponse?
    var ohttpStateMachine = OHTTPServerStateMachine()
    var requestTrackingID: String = ""

    init(
        attestationClient: CloudBoardAttestationAPIClientProtocol?,
        server: CloudBoardJobHelperAPIServerToClientProtocol,
        encodedRequestContinuation: AsyncStream<PipelinePayload<Data>>.Continuation,
        responseStream: AsyncStream<FinalizableChunk<Data>>,
        metrics: MetricsSystem
    ) {
        self.attestationClient = attestationClient
        self.server = server
        self.encodedRequestContinuation = encodedRequestContinuation
        self.responseStream = responseStream
        self.metrics = metrics
        self.laContext = LAContext()
    }

    deinit {
        // There is a chance that ``run`` never gets called from ``CloudBoardJobHelper`` in case an error is thrown
        // after instantiating ``CloudBoardMessenger`` and starting the XPC server to handle incoming workload requests
        // from cloudboardd but before ``CloudBoardMessenger.run`` is called.
        if case .awaitingKeys(let promise) = ohttpKeyState {
            promise.fail(with: CloudBoardMessengerError.runNeverCalled)
        }
    }

    public func invokeWorkloadRequest(_ request: InvokeWorkloadRequest) async throws {
        do {
            self.metrics.emit(Metrics.Messenger.TotalRequestsReceivedCounter(action: .increment))
            switch request {
            case .warmup(let warmupData):
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Received warmup data"
                ).log(to: Self.logger, level: .default)
                self.encodedRequestContinuation.yield(.warmup(warmupData))
            case .requestChunk(let encryptedPayload, let isFinal):
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Received request chunk"
                ).log(request: .init(chunk: encryptedPayload, isFinal: isFinal), to: Self.logger, level: .default)
                if let data = try self.ohttpStateMachine.receiveChunk(encryptedPayload, isFinal: isFinal) {
                    self.metrics.emit(Metrics.Messenger.RequestChunkReceivedSizeHistogram(value: data.count))
                    self.encodedRequestContinuation.yield(.chunk(.init(chunk: data, isFinal: isFinal)))
                }
            case .parameters(let parameters):
                self.requestTrackingID = parameters.requestID
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Received request parameters"
                ).log(to: Self.logger, level: .default)
                // We split up the parameters here into metadata to be processed within cb_jobhelper (one-time token)
                // and data to be forwarded to the cloud app (see ``ParametersData``)
                self.encodedRequestContinuation.yield(.oneTimeToken(parameters.oneTimeToken))
                self.encodedRequestContinuation.yield(.parameters(.init(parameters)))
                try await self.invokePrivateKeyRequest(
                    keyID: parameters.encryptedKey.keyID,
                    wrappedKey: parameters.encryptedKey.key
                )
                // rdar://126351696 (Swift compiler seems to ignore protocol conformances not used in the same target)
                _ = CryptoKitError.incorrectKeySize.publicDescription
            }
        } catch let error as ReportableJobHelperError {
            CloudBoardMessengerCheckpoint(
                logMetadata: self.logMetadata(),
                message: "invokeWorkloadRequest error",
                error: error
            ).log(to: Self.logger, level: .error)
            try await self.server.sendWorkloadResponse(WorkloadResponse.failureReport(error.reason))
            throw error.wrappedError
        } catch {
            CloudBoardMessengerCheckpoint(
                logMetadata: self.logMetadata(),
                message: "invokeWorkloadRequest error",
                error: error
            ).log(to: Self.logger, level: .error)
            throw error
        }
    }

    private func invokePrivateKeyRequest(keyID: Data, wrappedKey: Data) async throws {
        let keyIDEncoded = keyID.base64EncodedString()
        guard let ohttpKey = try await self.ohttpKeys.first(where: { $0.keyID == keyID }) else {
            let error = ReportableJobHelperError(
                wrappedError: NodeKeyError.unknownKeyID(keyIDEncoded),
                reason: .unknownKeyID
            )
            CloudBoardMessengerCheckpoint(
                logMetadata: self.logMetadata(),
                message: "No node key found with the provided key ID",
                error: error
            ).log(keyID: keyIDEncoded, to: Self.logger, level: .error)
            throw error
        }
        CloudBoardMessengerCheckpoint(
            logMetadata: self.logMetadata(),
            message: "Found attested key for request"
        ).log(keyID: keyIDEncoded, to: Self.logger, level: .info)

        guard ohttpKey.expiry >= .now else {
            let error = ReportableJobHelperError(
                wrappedError: NodeKeyError.expiredKey(keyIDEncoded),
                reason: .expiredKey
            )

            CloudBoardMessengerCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Provided key has expired",
                error: error
            ).log(keyID: keyIDEncoded, to: Self.logger, level: .error)

            throw error
        }

        do {
            let (chunks, encapsulator) = try self.ohttpStateMachine.receiveKey(
                wrappedKey,
                privateKey: ohttpKey.cachedKey
            )
            self.responseEncapsulator = encapsulator
            for chunk in chunks {
                self.encodedRequestContinuation.yield(.chunk(.init(chunk: chunk.chunk)))
            }
        }
    }

    func teardown() async throws {
        self.encodedRequestContinuation.yield(.teardown)
    }

    public func run() async throws {
        // Obtain initial set of SEP-backed node keys from the attestation daemon and register for key rotation
        // notifications
        let attestationClient: CloudBoardAttestationAPIClientProtocol = if self.attestationClient != nil {
            self.attestationClient!
        } else {
            await CloudBoardAttestationAPIXPCClient.localConnection()
        }
        // We need to register for key rotation events as keys might rotate while cb_jobhelper runs but before we
        // receive a request, in particular with prewarming enabled.
        await attestationClient.set(delegate: self)
        await attestationClient.connect()
        do {
            CloudBoardMessengerCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Requesting OHTTP node keys from attestation daemon"
            ).log(to: Self.logger, level: .default)
            let keySet = try await attestationClient.requestAttestedKeySet()
            CloudBoardMessengerCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Received OHTTP node keys from attestation daemon"
            ).log(attestedKeySet: keySet, to: Self.logger, level: .default)
            let keys = try Array(keySet: keySet, laContext: self.laContext)
            if case .awaitingKeys(let promise) = ohttpKeyState {
                promise.succeed(with: keys)
            }
            self.ohttpKeyState = .available(keys)

            var bufferedResponses = [FinalizableChunk<Data>]()
            var receivedFinal = false
            for await response in self.responseStream {
                defer { self.metrics.emit(Metrics.Messenger.TotalResponseChunksInBuffer(value: 0)) }
                self.metrics.emit(Metrics.Messenger.TotalResponseChunksReceivedCounter(action: .increment))
                self.metrics.emit(
                    Metrics.Messenger.TotalResponseChunkReceivedSizeHistogram(value: response.chunk.count)
                )
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Received response"
                ).log(response: response, to: Self.logger, level: .debug)
                // If we haven't received the decryption key and with that set up the encapsulator, we have to buffer
                // responses until we do
                if self.responseEncapsulator == nil {
                    CloudBoardMessengerCheckpoint(
                        logMetadata: self.logMetadata(),
                        message: "Buffering encoded response until OHTTP encapsulation is set up"
                    ).log(to: Self.logger, level: .default)
                    bufferedResponses.append(response)
                    self.metrics.emit(Metrics.Messenger.TotalResponseChunksInBuffer(value: bufferedResponses.count))
                    self.metrics.emit(
                        Metrics.Messenger
                            .TotalResponseChunksBufferedSizeHistogram(value: response.chunk.count)
                    )
                } else {
                    for response in bufferedResponses + [response] {
                        let responseMessage = try responseEncapsulator!.encapsulate(
                            response.chunk,
                            final: response.isFinal
                        )
                        CloudBoardMessengerCheckpoint(
                            logMetadata: self.logMetadata(),
                            message: "Sending encapsulated response"
                        ).log(response: response, to: Self.logger, level: .debug)
                        try await self.server.sendWorkloadResponse(.responseChunk(.init(
                            encryptedPayload: responseMessage,
                            isFinal: response.isFinal
                        )))
                        if response.isFinal {
                            receivedFinal = true
                        }
                        self.metrics.emit(Metrics.Messenger.TotalResponseChunksSentCounter(action: .increment))
                    }
                    bufferedResponses = []
                    self.metrics.emit(Metrics.Messenger.TotalResponseChunksInBuffer(value: 0))
                }
            }

            // Finish the request stream once the response stream has ended.
            self.encodedRequestContinuation.finish()

            if !receivedFinal {
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Encoded response stream finished without final response chunk"
                ).log(to: Self.logger, level: .error)
            } else {
                CloudBoardMessengerCheckpoint(
                    logMetadata: self.logMetadata(),
                    message: "Finished encoded response stream"
                ).log(to: Self.logger, level: .default)
            }
        } catch {
            CloudBoardMessengerCheckpoint(
                logMetadata: self.logMetadata(),
                message: "Error while running CloudBoardMessenger",
                error: error
            ).log(to: Self.logger, level: .fault)
            if case .awaitingKeys(let promise) = ohttpKeyState {
                promise.fail(with: error)
            }
            throw error
        }
    }

    func surpriseDisconnect() async {
        CloudBoardMessengerCheckpoint(
            logMetadata: self.logMetadata(),
            message: "Unexpectedly disconnected from cb_attestationd"
        ).log(to: Self.logger, level: .error)
    }

    func keyRotated(newKeySet: CloudBoardAttestationDAPI.AttestedKeySet) async throws {
        self.ohttpKeyState = try .available(Array(keySet: newKeySet, laContext: self.laContext))
    }
}

extension CloudBoardMessenger {
    private func logMetadata() -> CloudBoardJobHelperLogMetadata {
        return CloudBoardJobHelperLogMetadata(
            requestTrackingID: self.requestTrackingID
        )
    }
}

enum NodeKeyError: Error {
    case unknownKeyID(String)
    case expiredKey(String)
}

/// `ReportableJobHelperError` is  used to describe an error for which the underlying reason is reported back to
/// CloudBoard via a `WorkloadResponse`.
struct ReportableJobHelperError: Error {
    let wrappedError: Error
    let reason: FailureReason
}

struct CloudBoardMessengerCheckpoint: RequestCheckpoint {
    var requestID: String? {
        self.logMetadata.requestTrackingID
    }

    var operationName: StaticString

    var serviceName: StaticString = "cb_jobhelper"

    var namespace: StaticString = "cloudboard"

    var error: Error?

    var logMetadata: CloudBoardJobHelperLogMetadata

    var message: StaticString

    public init(
        logMetadata: CloudBoardJobHelperLogMetadata,
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

    public func log(request: FinalizableChunk<Data>, to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        requestChunkSize=\(request.chunk.count, privacy: .public)
        requestChunkIsFinal=\(request.isFinal)
        """)
    }

    public func log(response: FinalizableChunk<Data>, to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        responseChunkSize=\(response.chunk.count, privacy: .public)
        responseIsFinal=\(response.isFinal)
        """)
    }

    public func log(keyID: String, to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        keyID=\(keyID, privacy: .public)
        """)
    }

    public func log(attestedKeySet: AttestedKeySet, to logger: Logger, level: OSLogType = .default) {
        logger.log(level: level, """
        ttl=\(self.type, privacy: .public)
        jobID=\(self.logMetadata.jobID?.uuidString ?? "", privacy: .public)
        remotePid=\(String(describing: self.logMetadata.remotePID), privacy: .public)
        requestId=\(self.requestID ?? "", privacy: .public)
        tracing.name=\(self.operationName, privacy: .public)
        tracing.type=\(self.type, privacy: .public)
        service.name=\(self.serviceName, privacy: .public)
        service.namespace=\(self.namespace, privacy: .public)
        status=\(status?.rawValue ?? "", privacy: .public)
        error.type=\(self.error.map { String(describing: Swift.type(of: $0)) } ?? "", privacy: .public)
        error.description=\(self.error.map { String(reportable: $0) } ?? "", privacy: .public)
        error.detailed=\(self.error.map { String(describing: $0) } ?? "", privacy: .private)
        message=\(self.message, privacy: .public)
        keySet=\(attestedKeySet, privacy: .public)
        """)
    }
}

struct CachedAttestedKey {
    var attestedKey: AttestedKey
    var cachedKey: any HPKEDiffieHellmanPrivateKey

    init(_ attestedKey: AttestedKey, laContext: LAContext) throws {
        switch attestedKey.key {
        case .keychain(let persistentKeyReference):
            do {
                let secKey = try Keychain.fetchKey(persistentRef: persistentKeyReference)
                let cryptoKitKey = try SecureEnclave.Curve25519.KeyAgreement.PrivateKey(
                    from: secKey,
                    authenticationContext: laContext
                )
                self.cachedKey = cryptoKitKey
                CloudBoardMessenger.logger.debug("""
                message=\("Obtained OHTTP key from keychain", privacy: .public)
                publicKey=\(cryptoKitKey.publicKey.rawRepresentation.base64EncodedString(), privacy: .public)
                """)
            } catch {
                CloudBoardMessengerCheckpoint(
                    logMetadata: .init(),
                    message: "Failed to obtain OHTTP key from keychain",
                    error: error
                ).log(to: CloudBoardMessenger.logger, level: .fault)
                fatalError("Failed to obtain OHTTP key from keychain: \(String(reportable: error))")
            }
        case .direct(let inMemoryKey):
            self.cachedKey = try Curve25519.KeyAgreement.PrivateKey(rawRepresentation: inMemoryKey)
        }

        self.attestedKey = attestedKey
    }

    var keyID: Data {
        self.attestedKey.keyID
    }

    var expiry: Date {
        self.attestedKey.expiry
    }
}

extension [CachedAttestedKey] {
    init(keySet: AttestedKeySet, laContext: LAContext) throws {
        self = []
        self.reserveCapacity(keySet.unpublishedKeys.count + 1)

        try self.append(CachedAttestedKey(keySet.currentKey, laContext: laContext))

        for unpublishedKey in keySet.unpublishedKeys {
            try self.append(CachedAttestedKey(unpublishedKey, laContext: laContext))
        }
    }
}

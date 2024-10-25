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
import CloudBoardMetrics
import Foundation
import os

private typealias AttestationTimeToExpiryGauge = Metrics.AttestationProvider.AttestationTimeToExpiryGauge

/// Obtains and manages attestations provided by cb_attestationd
final actor AttestationProvider: CloudBoardAttestationAPIClientDelegateProtocol {
    private static let metricsRecordingInterval: Duration = .seconds(60)

    // This is shared with the state machine so can't be made private
    fileprivate static let logger: os.Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "AttestationProvider"
    )

    private let attestationClient: CloudBoardAttestationAPIClientProtocol
    private let metricsSystem: MetricsSystem
    private var connectionStateMachine: ConnectionStateMachine
    private var attestationStateMachine: AttestationStateMachine

    init(attestationClient: CloudBoardAttestationAPIClientProtocol, metricsSystem: MetricsSystem) {
        self.attestationClient = attestationClient
        self.metricsSystem = metricsSystem
        self.connectionStateMachine = ConnectionStateMachine()
        self.attestationStateMachine = AttestationStateMachine()
    }

    func run() async throws {
        await self.attestationClient.set(delegate: self)
        // Request attestation bundle to make it available as soon as possible
        _ = try await self.currentAttestationSet()

        while true {
            self.emitAttestationTimeToExpiry()
            try await Task.sleep(for: Self.metricsRecordingInterval)
        }
    }

    func attestationRotated(newAttestationSet: CloudBoardAttestationDAPI.AttestationSet) async throws {
        Self.logger
            .notice("Received new attestation set from attestation daemon: \(newAttestationSet, privacy: .public)")
        self.attestationStateMachine.attestationRotated(attestationSet: newAttestationSet)
    }

    func surpriseDisconnect() async {
        Self.logger.error("XPC connection to CloudBoard attestation daemon terminated unexpectedly")
    }

    func currentAttestationSet() async throws -> AttestationSet {
        try await self.ensureConnectionToAttestationDaemon()
        switch try self.attestationStateMachine.obtainAttestation() {
        case .requestAttestation:
            do {
                Self.logger.info("requesting attestation set from cb_attestationd")
                let attestationSet = try await self.attestationClient.requestAttestationSet()
                Self.logger.log("successfully obtained current attested set: \(attestationSet, privacy: .public)")
                return self.attestationStateMachine.attestationReceived(attestationSet: attestationSet)
            } catch {
                Self.logger
                    .error("failed to request attestation bundle: \(String(unredacted: error), privacy: .public)")
                return try self.attestationStateMachine.attestationRequestFailed(error: error)
            }
        case .waitForAttestation(let future):
            return try await future.valueWithCancellation
        case .continueWithAttestation(let attestationSet):
            return attestationSet
        }
    }

    func ensureConnectionToAttestationDaemon() async throws {
        switch self.connectionStateMachine.checkConnection() {
        case .connect:
            AttestationProvider.logger.info("connecting to cb_attestationd")
            await self.attestationClient.connect()
            self.connectionStateMachine.connectionEstablished()
        case .waitForConnection(let future):
            try await future.valueWithCancellation
        case .continue:
            // Nothing to do
            ()
        }
    }

    internal func emitAttestationTimeToExpiry() {
        let attestationAction = try? self.attestationStateMachine.obtainAttestation()
        switch attestationAction {
        case .continueWithAttestation(let attestationSet):
            self.metricsSystem.emit(
                AttestationTimeToExpiryGauge(expireAt: attestationSet.currentAttestation.expiry)
            )
        default:
            Self.logger.log("Attestation set not available yet. Skip emitting expiry metrics")
            return
        }
    }
}

private struct ConnectionStateMachine {
    internal enum ConnectionState: CustomStringConvertible {
        case initialized
        case connecting(Promise<Void, Never>)
        case connected

        var description: String {
            switch self {
            case .initialized:
                return "initialized"
            case .connecting:
                return "connecting"
            case .connected:
                return "connected"
            }
        }
    }

    private var state: ConnectionState

    init() {
        self.state = .initialized
    }

    enum ConnectAction {
        case connect
        case waitForConnection(Future<Void, Never>)
        case `continue`
    }

    mutating func checkConnection() -> ConnectAction {
        switch self.state {
        case .initialized:
            self.state = .connecting(Promise<Void, Never>())
            return .connect
        case .connecting(let promise):
            return .waitForConnection(Future(promise))
        case .connected:
            // Nothing to do, already connected
            return .continue
        }
    }

    mutating func connectionEstablished() {
        let state = self.state
        guard case .connecting(let promise) = state else {
            AttestationProvider.logger
                .error("unexpected connection state \(state, privacy: .public) after connecting to cb_attestationd")
            preconditionFailure(
                "unexpected connection state \(state) after connecting to cb_attestationd"
            )
        }
        promise.succeed()
        self.state = .connected
    }
}

private struct AttestationStateMachine {
    internal enum AttestationState: CustomStringConvertible {
        case initialized
        case awaitingAttestation(Promise<AttestationSet, Error>)
        case attestationAvailable(AttestationSet)
        case attestationUnavailable(Error)

        var description: String {
            switch self {
            case .initialized:
                return "initialized"
            case .awaitingAttestation:
                return "awaitingAttestation"
            case .attestationAvailable(let attestationSet):
                return "attestationAvailable(publicationExpiry: \(attestationSet.currentAttestation.publicationExpiry))"
            case .attestationUnavailable(let error):
                return "attestationUnavailable(error: \(error)"
            }
        }
    }

    private var state: AttestationState

    init() {
        self.state = .initialized
    }

    enum AttestationAction {
        case requestAttestation
        case waitForAttestation(Future<AttestationSet, Error>)
        case continueWithAttestation(AttestationSet)
    }

    mutating func obtainAttestation() throws -> AttestationAction {
        switch self.state {
        case .initialized:
            self.state = .awaitingAttestation(Promise<AttestationSet, Error>())
            return .requestAttestation
        case .awaitingAttestation(let promise):
            return .waitForAttestation(Future(promise))
        case .attestationAvailable(let attestationSet):
            return .continueWithAttestation(attestationSet)
        case .attestationUnavailable(let error):
            throw error
        }
    }

    mutating func attestationReceived(attestationSet: AttestationSet) -> AttestationSet {
        // We might have gotten additional requests or the key might have rotated and has been updated in the meantime
        switch self.state {
        case .awaitingAttestation(let promise):
            promise.succeed(with: attestationSet)
            self.state = .attestationAvailable(attestationSet)
            return attestationSet
        case .attestationAvailable(let attestationSet):
            // Key set has rotated in the meantime. Use the rotated set.
            return attestationSet
        case .initialized, .attestationUnavailable:
            // We should never get into any other state
            let state = self.state
            AttestationProvider.logger
                .error("unexpected state: \(state, privacy: .public) after requesting attestation set")
            preconditionFailure("unexpected state: \(state) after requesting attestation set")
        }
    }

    mutating func attestationRequestFailed(error: Error) throws -> AttestationSet {
        switch self.state {
        case .awaitingAttestation(let promise):
            promise.fail(with: error)
            self.state = .attestationUnavailable(error)
        case .attestationAvailable(let attestationSet):
            // Key has successfully rotated in the meantime. Use the rotated key and ignore error.
            AttestationProvider.logger.notice(
                "failed to request attestation set but attestation set has successfully rotated in the meantime. Continuing with rotated attestation set."
            )
            return attestationSet
        case .initialized, .attestationUnavailable:
            // We should never get into any other state
            let state = self.state
            AttestationProvider.logger
                .error("unexpected state: \(state, privacy: .public) when handling attestation request error")
            preconditionFailure("unexpected state: \(state) when handling attestation request error")
        }
        // Rethrow if we couldn't recover from the error
        throw error
    }

    mutating func attestationRotated(attestationSet: AttestationSet) {
        switch self.state {
        case .awaitingAttestation(let promise):
            promise.succeed(with: attestationSet)
        case .initialized, .attestationAvailable, .attestationUnavailable:
            // Nothing to do
            ()
        }
        self.state = .attestationAvailable(attestationSet)
    }
}

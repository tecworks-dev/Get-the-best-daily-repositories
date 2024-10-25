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

// Copyright © 2023 Apple. All rights reserved.

import CloudBoardAsyncXPC
import CloudBoardAttestationDAPI
import CloudBoardCommon
import CloudBoardMetrics
import Foundation
import NIOCore
import os

/// Serves requests from other components (cloudboardd and cb_jobhelper) for attestations/attested keys
actor CloudBoardAttestationServer: CloudBoardAttestationAPIServerDelegateProtocol {
    public static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "CloudBoardAttestationServer"
    )

    private let apiServer: CloudBoardAttestationAPIServerProtocol
    private let attestationProvider: AttestationProvider
    private let keyLifetime: TimeAmount
    private let keyExpiryGracePeriod: TimeAmount
    private let keychain: SecKeychain?

    private let metrics: MetricsSystem

    private var stateMachine: AttestationStateMachine
    private var _unpublishedKeys: [AttestedKey] = []

    private var activeUnpublishedKeys: [AttestedKey] {
        return self._unpublishedKeys.filter { $0.expiry > .now }
    }

    init(
        apiServer: CloudBoardAttestationAPIServerProtocol,
        attestationProvider: AttestationProvider,
        keyLifetime: TimeAmount,
        keyExpiryGracePeriod: TimeAmount,
        keychain: SecKeychain? = nil,
        metrics: MetricsSystem
    ) {
        self.apiServer = apiServer
        self.attestationProvider = attestationProvider
        self.keyLifetime = keyLifetime
        self.keyExpiryGracePeriod = keyExpiryGracePeriod
        self.keychain = keychain
        self.stateMachine = AttestationStateMachine()
        self.metrics = metrics
    }

    func requestAttestedKeySet() async throws -> AttestedKeySet {
        Self.logger.info("Received request for attested key set")
        do {
            let activeKey = try await self.obtainAttestedKey()
            let keySet = AttestedKeySet(currentKey: activeKey, unpublishedKeys: self.activeUnpublishedKeys)
            Self.logger.log("Returning key set: \(keySet, privacy: .public)")
            return keySet
        } catch {
            Self.logger.error("Failed to obtain attested key: \(String(unredacted: error), privacy: .public)")
            throw error
        }
    }

    func requestAttestationSet() async throws -> AttestationSet {
        Self.logger.info("Received request for attestation set")
        do {
            let activeKey = try await self.obtainAttestedKey()
            let attestationSet = AttestationSet(
                currentAttestation: .init(key: activeKey),
                unpublishedAttestations: self.activeUnpublishedKeys.map { .init(key: $0) }
            )
            Self.logger.log("Returning attestation set: \(attestationSet, privacy: .public)")
            return attestationSet
        } catch {
            Self.logger.error("Failed to obtain attested key: \(String(unredacted: error), privacy: .public)")
            throw error
        }
    }

    private func obtainAttestedKey() async throws -> AttestedKey {
        switch try self.stateMachine.obtainAttestedKey() {
        case .createAttestedKey:
            do {
                Self.logger.info("Requested to create new attested key")
                let attestedKey = try await createNewAttestedKey()
                return self.stateMachine.attestedKeyReceived(key: attestedKey)
            } catch {
                Self.logger.error("Failed to create attested key: \(String(unredacted: error), privacy: .public)")
                return try self.stateMachine.keyRequestFailed(error: error)
            }
        case .waitForAttestedKey(let future):
            Self.logger.info("Waiting for attested key to become available")
            return try await future.valueWithCancellation
        case .continueWithAttestedKey(let key):
            Self.logger.info("Requested to continue with key \(key, privacy: .public)")
            return key
        }
    }

    private func createNewAttestedKey() async throws -> AttestedKey {
        // Actual key expiry is slightly longer than the advertised key expiry to avoid TOCTOU issues and to
        // allow for latency between client-side validation and CloudBoard receiving requests
        let now = Date.now
        let advertisedKeyExpiry = now + self.keyLifetime.timeInterval
        let keyExpiry = advertisedKeyExpiry + self.keyExpiryGracePeriod.timeInterval
        // Attestations should be published by ROPES for about half their lifetime plus a grace period allowing us to
        // ensure that by the time ROPES fetches attestations the next time we have rotated the key/attestation
        var publicationExpiry = now + self.keyLifetime.timeInterval / 2.0
        publicationExpiry += self.keyExpiryGracePeriod.timeInterval

        let internalAttestedKey = try await self.attestationProvider
            .createAttestedKey(attestationBundleExpiry: advertisedKeyExpiry)
        return try internalAttestedKey.exportable(
            expiry: keyExpiry,
            publicationExpiry: publicationExpiry,
            keychain: self.keychain
        )
    }

    func run() async throws {
        await self.apiServer.set(delegate: self)
        await self.apiServer.connect()

        // Delete existing keys
        self.deleteExistingKeys()

        // Create initial attested key
        Self.logger.info("Creating initial attested key")
        var currentKey = try await self.obtainAttestedKey()

        while true {
            // We substract the grace period as we want to rotate the key once the previous key has reached half of the
            // time of the advertised expiry, having ~2 overlapping keys active at a time by default
            let currentKeyHalfLife = (currentKey.expiry.timeIntervalSinceNow - self.keyExpiryGracePeriod.timeInterval) /
                2.0
            Self.logger
                .notice("Sleeping for \(currentKeyHalfLife, privacy: .public) seconds before rotating the attested key")
            try await Task.sleep(for: .seconds(currentKeyHalfLife))

            let newKey = try await createNewAttestedKey()
            self.stateMachine.keyRotated(key: newKey)
            self._unpublishedKeys += [currentKey]
            currentKey = newKey

            // Remove expired keys from keychain
            self._unpublishedKeys = self._unpublishedKeys.filter { key in
                if key.expiry < .now {
                    Self.logger.notice(
                        "Removing expired key with key ID \(key.keyID.base64EncodedString(), privacy: .public) and expiry \(key.expiry, privacy: .public)"
                    )
                    // Remove from keychain if persisted
                    do {
                        switch key.key {
                        case .direct:
                            // Nothing to do
                            ()
                        case .keychain(let persistentKeyRef):
                            try Keychain.delete(persistentKeyRef: persistentKeyRef, keychain: self.keychain)
                        }
                    } catch {
                        Self.logger.error(
                            "Failed to delete key with key ID \(key.keyID.base64EncodedString(), privacy: .public): \(String(unredacted: error), privacy: .public)"
                        )
                    }
                    return false
                } else {
                    return true
                }
            }

            let keySet = AttestedKeySet(currentKey: newKey, unpublishedKeys: self.activeUnpublishedKeys)
            self.metrics.emit(Metrics.CloudBoardAttestationServer.KeyRotationCounter(action: .increment(by: 1)))
            Self.logger.notice("Broadcasting new attested key set: \(keySet, privacy: .public)")
            try await self.apiServer.keyRotated(newKeySet: keySet)
            try await self.apiServer.attestationRotated(newAttestationSet: .init(keySet: keySet))
        }
    }

    private func deleteExistingKeys() {
        do {
            let keys = try Keychain.findKeys(for: Keychain.baseNodeKeyQuery, keychain: self.keychain)
            Self.logger.log("Found \(keys.count, privacy: .public) existing key(s)")
            for key in keys {
                let keyAttributes = SecKeyCopyAttributes(key)
                do {
                    try Keychain.delete(key: key, keychain: self.keychain)
                    Self.logger.log("Successfully deleted old key: \(keyAttributes, privacy: .public)")
                } catch {
                    Self.logger.error(
                        "Failed to delete key with attributes \(keyAttributes, privacy: .public): \(String(unredacted: error), privacy: .public)"
                    )
                }
            }
        } catch {
            Self.logger.error(
                "Failed to query for existing keys: \(String(unredacted: error), privacy: .public)"
            )
        }
    }
}

private struct AttestationStateMachine {
    internal enum AttestationState: CustomStringConvertible {
        case initialized
        case awaitingAttestedKey(Promise<AttestedKey, Error>)
        case attestedKeyAvailable(key: AttestedKey)
        case attestedKeyUnavailable(Error)

        var description: String {
            switch self {
            case .initialized:
                return "initialized"
            case .awaitingAttestedKey:
                return "awaitingAttestedKey"
            case .attestedKeyAvailable(let key):
                return "attestationAvailable(expiry: \(key.expiry))"
            case .attestedKeyUnavailable(let error):
                return "attestationUnavailable(error: \(error)"
            }
        }
    }

    private var state: AttestationState

    init() {
        self.state = .initialized
    }

    enum AttestedKeyAction {
        case createAttestedKey
        case waitForAttestedKey(Future<AttestedKey, Error>)
        case continueWithAttestedKey(AttestedKey)
    }

    mutating func obtainAttestedKey() throws -> AttestedKeyAction {
        switch self.state {
        case .initialized:
            self.state = .awaitingAttestedKey(Promise<AttestedKey, Error>())
            return .createAttestedKey
        case .awaitingAttestedKey(let promise):
            return .waitForAttestedKey(Future(promise))
        case .attestedKeyAvailable(let key):
            return .continueWithAttestedKey(key)
        case .attestedKeyUnavailable(let error):
            throw error
        }
    }

    mutating func attestedKeyReceived(key: AttestedKey) -> AttestedKey {
        // We might have gotten additional requests or the key might have rotated and has been updated in the meantime
        switch self.state {
        case .awaitingAttestedKey(let promise):
            promise.succeed(with: key)
            self.state = .attestedKeyAvailable(key: key)
            return key
        case .attestedKeyAvailable(let key):
            // Key has rotated in the meantime. Use the rotated key.
            return key
        case .initialized, .attestedKeyUnavailable:
            // We should never get into any other state
            let state = self.state
            CloudBoardAttestationServer.logger
                .error("unexpected state: \(state, privacy: .public) after requesting attested key")
            preconditionFailure("unexpected state: \(state) after requesting attested key")
        }
    }

    mutating func keyRotated(key: AttestedKey) {
        switch self.state {
        case .awaitingAttestedKey(let promise):
            promise.succeed(with: key)
        case .initialized, .attestedKeyAvailable, .attestedKeyUnavailable:
            // do nothing
            ()
        }
        self.state = .attestedKeyAvailable(key: key)
    }

    mutating func keyRequestFailed(error: Error) throws -> AttestedKey {
        switch self.state {
        case .awaitingAttestedKey(let promise):
            promise.fail(with: error)
            self.state = .attestedKeyUnavailable(error)
            // Rethrow since we couldn't recover from the error
            throw error
        case .attestedKeyAvailable(let key):
            // Key has successfully rotated in the meantime. Use the rotated key and ignore error.
            CloudBoardAttestationServer.logger.warning(
                "failed to request attestaton but attestation has successfully rotated in the meantime. Continuing with rotated attestation."
            )
            return key
        case .initialized, .attestedKeyUnavailable:
            // We should never get into any other state
            let state = self.state
            CloudBoardAttestationServer.logger
                .error("unexpected state: \(state, privacy: .public) when handling attestation request error")
            preconditionFailure("unexpected state: \(state) when handling attestation request error")
        }
    }
}

extension InternalAttestedKey {
    /// Creates AttestedKey from the InternalAttestedKey. For a SEP-backed key this will store the key in the keychain
    /// and obtain a persistent reference that can be shared across process boundaries.
    func exportable(expiry: Date, publicationExpiry: Date, keychain: SecKeychain?) throws -> AttestedKey {
        let key: AttestedKeyType
        switch self.key {
        case .direct(let data):
            key = .direct(privateKey: data)
        case .sepKey(let secKey):
            try Keychain.add(key: secKey, keychain: keychain)
            do {
                key = try .keychain(persistentKeyReference: secKey.persistentRef())
            } catch {
                CloudBoardAttestationServer.logger.error(
                    "Failed to obtain persistent reference for node key: \(error)"
                )
                throw error
            }
        }
        return AttestedKey(
            key: key,
            attestationBundle: self.attestationBundle,
            expiry: expiry,
            publicationExpiry: publicationExpiry
        )
    }
}

extension Duration {
    public init(_ timeAmount: TimeAmount) {
        self = .nanoseconds(timeAmount.nanoseconds)
    }
}

extension TimeAmount {
    public var timeInterval: TimeInterval {
        return Double(self.nanoseconds) / 1_000_000_000
    }
}

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
//  EnsembleChannelSecurity.swift
//  CloudAttestation
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import CryptoKit
@_weakLinked import SecureConfigDB
import os.log

/// An umberella namespace for establishing secure communication links between Compute Ensemble nodes.
public enum EnsembleChannelSecurity {
    public typealias UDID = String
    internal typealias ChassisID = String
    internal typealias CertificateData = Data

    @available(*, deprecated)
    public static let sealedHashSlotUUID = UUID(uuidString: "B6FF3FC5-7CAF-4898-82F5-4887ED946ABC")!
    @available(*, deprecated)
    public static let sealedHashSlotSalt = "Salt"
}

// MARK: - Leader

extension EnsembleChannelSecurity {
    public struct Leader {
        static let logger = Logger(subsystem: "com.apple.CloudAttestation", category: "EnsembleChannelSecurity.Leader")

        var udid: UDID
        var followerUDIDs: Set<UDID>
        var leaderHPKE: EnsembleHPKE.Leader
        var _symmetricKey: CryptoKit.SymmetricKey
        var fingerprints: [Data]?
        var topology: [ChassisID: Set<UDID>] = [:]
        var validateTopology: Bool { fingerprints != nil }

        /// The symmetric key that is intended to be distributed to all follower nodes as a shared secret for further KDFs.
        public var symmetricKey: CryptoKit.SymmetricKey { _symmetricKey }
        /// The max number of chassis allowed for an Ensemble. Defaults to 2.
        public var maxChassisCount: Int = 2
        /// The max number of nodes per chassis allowed for an Ensemble. Defaults to 4.
        public var maxNodesPerChassis: Int = 4

        /// Creates a new leader object.
        /// - Parameters:
        ///   - udid: The UDID of the leader node.
        ///   - followerUDIDs: The UDIDs of the follower nodes.
        ///   - keySize: The size of the symmetric key to be generated. Defaults to .bits256.
        public init(udid: UDID, followerUDIDs: some Sequence<UDID>, keySize: SymmetricKeySize = .bits256) throws {
            guard #_hasSymbol(SCDataBase.self) else {
                throw Error.noSecureConfigDB
            }

            self = try .init(
                udid: udid,
                followerUDIDs: followerUDIDs,
                leaderHPKE: try EnsembleHPKE.Leader(),
                keySize: keySize
            )
        }

        init(
            udid: UDID,
            followerUDIDs: some Sequence<UDID>,
            leaderHPKE: EnsembleHPKE.Leader,
            keySize: SymmetricKeySize
        ) throws {
            self.udid = udid
            self.followerUDIDs = Set(followerUDIDs)
            self.leaderHPKE = leaderHPKE
            self._symmetricKey = SymmetricKey(size: keySize)
            self.fingerprints = Self.readEnsembleFingerprints(using: leaderHPKE.attestor.assetProvider)
            if self.validateTopology {
                try self.insertSelfIntoTopology()
            }
        }

        /// Returns the leader's attestation.
        public mutating func attest() async throws -> AttestationBundle {
            let attestation = try await self.leaderHPKE.attest()
            Self.logger.log("Leader attestation generated")
            return attestation
        }

        /// Performs a rekey, discarding any HPKE asymmetric keys.
        public mutating func rekey() throws {
            try self.leaderHPKE.rekey()
            self._symmetricKey = SymmetricKey(size: SymmetricKeySize(bitCount: _symmetricKey.bitCount))
            Self.logger.log("Leader rekeyed")
        }

        /// Initiates pairing with another device authenticated by their attestation. Calls ``EnsembleChannelSecurity/Leader/rekey()`` to prevent key reuse
        /// - Parameters:
        ///   - followerUDID: The follower's UDID.
        ///   - attestation: The follower's attestation bundle.
        /// - Returns: a ``PairingData`` object containing HPKE cipherText and encapsulated key.
        public mutating func pair(with followerUDID: UDID, authenticatedBy attestation: AttestationBundle) async throws -> PairingData {
            Self.logger.log("Leader pairing with \(followerUDID, privacy: .public)")

            guard self.followerUDIDs.contains(followerUDID) else {
                let followerUDIDs = self.followerUDIDs
                Self.logger.log("Follower UDID \(followerUDID, privacy: .public) not in expected set \(followerUDIDs, privacy: .public)")
                throw Error.unexpectedUDID(followerUDID)
            }

            var sender: HPKE.Sender
            if self.validateTopology {
                guard let fingerprints else {
                    throw Error.missingEnsembleMemberFingerprints
                }

                let policy = self.leaderHPKE.validator.policyFor(udid: followerUDID, fingerprints: fingerprints)

                let (hpkeSender, validatedBundle) = try await leaderHPKE.sender(for: attestation, policy: policy)
                guard let leafCertData = validatedBundle.provisioningCertificateChain.first else {
                    throw Error.provisioningCertificateError(.missingCertificate)
                }

                let provisioningCertificate = try ProvisioningCertificate(data: leafCertData)
                guard let certIdentityString = provisioningCertificate.deviceIdentity?.identity else {
                    throw Error.provisioningCertificateError(.missingIdentity)
                }

                guard let certIdentity = SEP.Identity(string: certIdentityString) else {
                    throw Error.provisioningCertificateError(.malformedIdentity)
                }

                guard let certChassisID: ChassisID = provisioningCertificate.chassisID?.string else {
                    throw Error.provisioningCertificateError(.missingChassisID)
                }

                guard certIdentity.udid == followerUDID else {
                    Self.logger.log("Follower UDID \(followerUDID, privacy: .public) does not match provisioning certificate UDID \(certIdentity.udid , privacy: .public)")
                    throw Error.provisioningCertificateError(.mismatchingUDID(provided: followerUDID, certificate: certIdentity.udid))
                }

                try updateTopology(chassisID: certChassisID, udid: certIdentity.udid)

                sender = hpkeSender
            } else {
                let policy = self.leaderHPKE.validator.policyFor(udid: followerUDID)

                let (hpkeSender, _) = try await leaderHPKE.sender(for: attestation, policy: policy)

                sender = hpkeSender
            }

            let cipherText = try self.symmetricKey.withUnsafeBytes { ptr in
                try sender.seal(ArraySlice(ptr.assumingMemoryBound(to: UInt8.self)))
            }

            // always rekey the HPKE
            try self.leaderHPKE.rekey()

            return PairingData(cipherText: cipherText, encapsulatedKey: sender.encapsulatedKey)
        }

        private mutating func updateTopology(chassisID: ChassisID, udid: UDID) throws {
            if var nodesByChassis = self.topology[chassisID] {
                nodesByChassis.insert(udid)
                self.topology[chassisID] = nodesByChassis
            } else {
                self.topology[chassisID] = [udid]
            }

            guard self.topology.count <= self.maxChassisCount else {
                throw Error.topologyError(.tooManyChassis(count: self.topology.count, maximum: self.maxChassisCount))
            }

            try self.topology.forEach { (key: EnsembleChannelSecurity.ChassisID, value: Set<EnsembleChannelSecurity.UDID>) in
                guard value.count <= self.maxNodesPerChassis else {
                    throw Error.topologyError(.tooManyNodesInChassis(chassis: chassisID, count: value.count, maximum: self.maxNodesPerChassis))
                }
            }
        }

        private static func readEnsembleFingerprints(using assetProvider: some AttestationAssetProvider) -> [Data]? {
            guard let secureConfigSlot = try? assetProvider.sealedHashEntries[secureConfigSlotUUID] else {
                Self.logger.warning("No SecureConfig entries in SecureConfigDB")
                return nil
            }

            let secureConfigs = secureConfigSlot.lazy.filter { $0.data != nil }.compactMap { SecureConfig(from: $0.data!) }
            guard let darwinInitConfig = secureConfigs.first(where: { $0.name == "darwin-init" && $0.mimeType == "application/json" }) else {
                Self.logger.warning("No darwin-init entry in SecureConfigDB")
                return nil
            }
            do {
                let darwinInit = try DarwinInit(from: darwinInitConfig)
                return darwinInit.ensembleCertificateFingerprints
            } catch {
                Self.logger.error("Malformed darwin-init: \(error, privacy: .public)")
                return nil
            }
        }

        private mutating func insertSelfIntoTopology() throws {
            guard let leaf = try self.leaderHPKE.attestor.assetProvider.provisioningCertificateChain.first else {
                throw Error.provisioningCertificateError(.missingCertificate)
            }
            let cert = try ProvisioningCertificate(data: leaf)
            guard let chassisID = cert.chassisID?.string else {
                throw Error.provisioningCertificateError(.missingChassisID)
            }
            guard let identityString = cert.deviceIdentity?.identity else {
                throw Error.provisioningCertificateError(.missingIdentity)
            }
            guard let identity = SEP.Identity(string: identityString) else {
                throw Error.provisioningCertificateError(.malformedIdentity)
            }
            try self.updateTopology(chassisID: chassisID, udid: identity.udid)
        }
    }
}

// MARK: - Follower
extension EnsembleChannelSecurity {
    public struct Follower {
        static let logger = Logger(subsystem: "com.apple.CloudAttestation", category: "EnsembleChannelSecurity.Leader")

        var udid: UDID
        var leaderUDID: UDID
        var followerHPKE: EnsembleHPKE.Follower

        /// Creates a new follower.
        /// - Parameters:
        ///   - udid: The UDID of the follower.
        ///   - leaderUDID: The UDID of the leader.
        public init(udid: UDID, leaderUDID: UDID) throws {
            self = .init(udid: udid, leaderUDID: leaderUDID, followerHPKE: try EnsembleHPKE.Follower())
        }

        @_spi(Private)
        public init(udid: UDID, leaderUDID: UDID, followerHPKE: EnsembleHPKE.Follower) {
            self.udid = udid
            self.leaderUDID = leaderUDID
            self.followerHPKE = followerHPKE
        }

        /// Returns the current follower's attestation.
        public func attest() async throws -> AttestationBundle {
            return try await self.followerHPKE.attest()
        }

        @_spi(Private)
        public mutating func rekey() throws {
            try self.followerHPKE.rekey()
            Self.logger.log("Follower rekeyed")
        }

        /// Completes pairing with a Leader node and returns the shared symmetric key as a result.
        /// - Parameters:
        ///   - pairingData: The pairing data.
        ///   - attestation: The attestation bundle.
        /// - Returns: the shared symmetric key.
        public mutating func completePairing(using pairingData: PairingData, authenticatedBy attestation: AttestationBundle) async throws -> CryptoKit.SymmetricKey {
            let leaderUDID = self.leaderUDID
            let policy = self.followerHPKE.validator.policyFor(udid: leaderUDID)

            var (recipient, _) = try await self.followerHPKE.recipient(for: attestation, encapsulatedKey: pairingData.encapsulatedKey, policy: policy)

            let symKey = try CryptoKit.SymmetricKey(data: recipient.open(pairingData.cipherText))

            // discard ephemeral key after use
            try self.rekey()

            Self.logger.log("Follower completed pairing with leader: \(leaderUDID, privacy: .public)")

            return symKey
        }
    }
}

// MARK: - PairingData
extension EnsembleChannelSecurity {
    public struct PairingData: Hashable, Codable, Sendable {
        public let cipherText: Data
        public let encapsulatedKey: Data
    }
}

// MARK: - Error API
extension EnsembleChannelSecurity {
    public enum Error: Swift.Error {
        case unexpectedUDID(UDID)
        case noSecureConfigDB
        case malformedSecureConfigEntry(Swift.Error)
        case missingEnsembleMemberFingerprints
        case provisioningCertificateError(ProvisioningCertificateError)
        case topologyError(TopologyError)

        public enum ProvisioningCertificateError {
            case missingCertificate
            case missingIdentity
            case malformedIdentity
            case missingChassisID
            case mismatchingUDID(provided: UDID, certificate: UDID)
        }

        public enum TopologyError {
            case tooManyChassis(count: Int, maximum: Int)
            case tooManyNodesInChassis(chassis: UDID, count: Int, maximum: Int)
        }
    }
}

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
//  EnsembleHPKEKeyAgreement.swift
//  CloudAttestation
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import Foundation
import CryptoKit
import CryptoKitPrivate
import Security_Private.SecKeyPriv

@_spi(Private)
public enum EnsembleHPKE {
    public static let ciphersuite = HPKE.Ciphersuite.P256_SHA256_AES_GCM_256

    static func createSEPKey() throws -> SecKey {
        let acl = SecAccessControlCreateWithFlags(nil, kSecAttrAccessibleAlwaysThisDeviceOnlyPrivate, SecAccessControlCreateFlags.privateKeyUsage, nil)!

        let keyAttr: [CFString: Any] = [
            kSecAttrKeyType: kSecAttrKeyTypeECSECPrimeRandom as CFString,
            kSecAttrKeySizeInBits: 256 as CFNumber,
            kSecAttrTokenID: kSecAttrTokenIDSecureEnclave,
            kSecAttrIsPermanent: false,
            kSecPrivateKeyAttrs: [
                kSecKeyOSBound: true,
                kSecKeySealedHashesBound: true,
                kSecAttrAccessControl: acl,
            ] as CFDictionary,
        ]

        var error: Unmanaged<CFError>?
        let key = SecKeyCreateRandomKey(keyAttr as CFDictionary, &error)
        if let error = error {
            throw error.takeRetainedValue()
        }

        return key!
    }

    static func createSoftwareKey() throws -> SecKey {
        let privateKey = P256.KeyAgreement.PrivateKey()
        let attributes =
            [
                kSecAttrKeyType: kSecAttrKeyTypeECSECPrimeRandom as CFString,
                kSecAttrKeyClass: kSecAttrKeyClassPrivate,
                kSecAttrIsPermanent: false,
            ] as [String: Any]
        var error: Unmanaged<CFError>?
        guard let secKey = SecKeyCreateWithData(privateKey.x963Representation as CFData, attributes as CFDictionary, &error) else {
            throw error!.takeRetainedValue()
        }
        return secKey
    }
}

// MARK: - Leader
extension EnsembleHPKE {
    public struct Leader: Sendable {
        private var keyAgreementKey: SendableSecKey
        var validator: EnsembleValidator
        var attestor: EnsembleAttestor
        var softwareKey: Bool

        /// Creates a new leader HPKE object.
        public init() throws {
            self = try .init(attestor: EnsembleAttestor(), validator: EnsembleValidator())
        }

        /// Creates a new leader HPKE object.
        /// - Parameters:
        ///   - attestor: The attestor.
        ///   - validator: The validator.
        public init(attestor: EnsembleAttestor, validator: EnsembleValidator) throws {
            self.softwareKey = false
            self.keyAgreementKey = SendableSecKey(secKey: try EnsembleHPKE.createSEPKey())
            self.validator = validator
            self.attestor = attestor
        }

        init(attestor: EnsembleAttestor, validator: EnsembleValidator, softwareKey: Bool) throws {
            self.softwareKey = softwareKey
            self.keyAgreementKey = SendableSecKey(secKey: try softwareKey ? createSoftwareKey() : createSEPKey())
            self.validator = validator
            self.attestor = attestor
        }

        /// Returns the attestation bundle for the Leader.
        public func attest() async throws -> AttestationBundle {
            try await self.attestor.attest(key: self.keyAgreementKey.secKey)
        }

        /// Performs a rekey.
        public mutating func rekey() throws {
            self.keyAgreementKey = SendableSecKey(secKey: try softwareKey ? createSoftwareKey() : createSEPKey())
        }

        /// Returns an HPKE sender object for the given recipient, authenticated by their attestation bundle.
        /// - Parameters:
        ///   - recipient: The recipient.
        ///   - info: The recipient.
        public func sender(for recipient: AttestationBundle, info: some DataProtocol = Data()) async throws -> (HPKE.Sender, Validated.AttestationBundle) {
            try await self.sender(for: recipient, info: info, policy: self.validator.defaultPolicy)
        }

        /// Returns an HPKE sender object for the given recipient, authenticated by their attestation bundle.
        /// - Parameters:
        ///   - recipient: The recipient.
        ///   - info: The recipient.
        ///   - policy: The attestation policy.
        public func sender(
            for recipient: AttestationBundle,
            info: some DataProtocol = Data(),
            policy: some AttestationPolicy
        ) async throws -> (HPKE.Sender, Validated.AttestationBundle) {
            let (peerPubKeyData, expiration, validated) = try await self.validator.validate(bundle: recipient, policy: policy)

            guard expiration >= Date.now else {
                throw EnsembleHPKE.Error.expiredAttestation
            }

            let peerPubKey =
                switch peerPubKeyData {
                case .x963(let data):
                    try P256.KeyAgreement.PublicKey(x963Representation: data)
                case .curve25519(_):
                    throw Error.unsupportedKeyAlgorithm
                }
            guard softwareKey else {
                let privateKey = try SecureEnclave.P256.KeyAgreement.PrivateKey(from: self.keyAgreementKey.secKey)
                return (try HPKE.Sender(recipientKey: peerPubKey, ciphersuite: EnsembleHPKE.ciphersuite, info: Data(info), authenticatedBy: privateKey), validated)
            }
            var error: Unmanaged<CFError>?
            guard let x963 = SecKeyCopyExternalRepresentation(self.keyAgreementKey.secKey, &error) else {
                throw error!.takeRetainedValue()
            }
            let privateKey = try P256.KeyAgreement.PrivateKey(x963Representation: x963 as Data)
            return (try HPKE.Sender(recipientKey: peerPubKey, ciphersuite: EnsembleHPKE.ciphersuite, info: Data(info), authenticatedBy: privateKey), validated)
        }
    }
}

// MARK: - Follower
extension EnsembleHPKE {
    public struct Follower: Sendable {
        private var keyAgreementKey: SendableSecKey
        var validator: EnsembleValidator
        var attestor: Attestor
        var softwareKey: Bool

        /// Creates a new ``EnsembleHPKE/Follower``.
        public init() throws {
            self = try .init(attestor: EnsembleAttestor(), validator: EnsembleValidator())
        }

        /// Creates a new ``EnsembleHPKE/Follower``.
        /// - Parameters:
        ///   - attestor: The `EnsembleAttestor`.
        ///   - validator: The `EnsembleValidator`.
        public init(attestor: EnsembleAttestor, validator: EnsembleValidator) throws {
            self.softwareKey = false
            self.keyAgreementKey = SendableSecKey(secKey: try createSEPKey())
            self.validator = validator
            self.attestor = attestor
        }

        init(attestor: EnsembleAttestor, validator: EnsembleValidator, softwareKey: Bool) throws {
            self.softwareKey = softwareKey
            self.keyAgreementKey = SendableSecKey(secKey: try softwareKey ? createSoftwareKey() : createSEPKey())
            self.validator = validator
            self.attestor = attestor
        }

        /// Performs a rekey.
        public mutating func rekey() throws {
            self.keyAgreementKey = SendableSecKey(secKey: try softwareKey ? createSoftwareKey() : createSEPKey())
        }

        /// Returns the folloer's ``AttestationBundle``
        public func attest() async throws -> AttestationBundle {
            try await self.attestor.attest(key: self.keyAgreementKey.secKey)
        }

        /// Returns an an `HPKE.Recipient` that authenticates against the leader's attestation bundle.
        /// - Parameters:
        ///   - sender: The ``AttestationBundle`` of the sender.
        ///   - encapsulatedKey: The ``EncapsulatedKey``.
        ///   - info: The ``AttestationBundle``'s ``info``.
        public func recipient(for sender: AttestationBundle, encapsulatedKey: Data, info: some DataProtocol = Data()) async throws -> (HPKE.Recipient, Validated.AttestationBundle)
        {
            try await self.recipient(for: sender, encapsulatedKey: encapsulatedKey, info: info, policy: self.validator.defaultPolicy)
        }

        /// Returns an an `HPKE.Recipient` that authenticates against the leader's attestation...
        /// - Parameters:
        ///   - sender: The ``AttestationBundle`` of the sender.
        ///   - encapsulatedKey: The ``EncapsulatedKey``.
        ///   - info: The ``AttestationBundle``'s ``info``.
        ///   - policy: The ``AttestationPolicy``.
        public func recipient(
            for sender: AttestationBundle,
            encapsulatedKey: Data,
            info: some DataProtocol = Data(),
            policy: some AttestationPolicy
        ) async throws -> (HPKE.Recipient, Validated.AttestationBundle) {
            let (peerPubKeyData, expiration, validated) = try await self.validator.validate(bundle: sender, policy: policy)

            guard expiration >= Date.now else {
                throw EnsembleHPKE.Error.expiredAttestation
            }

            let peerPubKey =
                switch peerPubKeyData {
                case .x963(let data):
                    try P256.KeyAgreement.PublicKey(x963Representation: data)
                case .curve25519(_):
                    throw Error.unsupportedKeyAlgorithm
                }

            guard softwareKey else {
                let privateKey = try SecureEnclave.P256.KeyAgreement.PrivateKey(from: self.keyAgreementKey.secKey)
                return (
                    try HPKE.Recipient(
                        privateKey: privateKey,
                        ciphersuite: EnsembleHPKE.ciphersuite,
                        info: Data(info),
                        encapsulatedKey: encapsulatedKey,
                        authenticatedBy: peerPubKey
                    ),
                    validated
                )
            }
            var error: Unmanaged<CFError>?
            guard let x963 = SecKeyCopyExternalRepresentation(self.keyAgreementKey.secKey, &error) else {
                throw error!.takeRetainedValue()
            }
            let privateKey = try P256.KeyAgreement.PrivateKey(x963Representation: x963 as Data)
            return (
                try HPKE.Recipient(privateKey: privateKey, ciphersuite: EnsembleHPKE.ciphersuite, info: Data(info), encapsulatedKey: encapsulatedKey, authenticatedBy: peerPubKey),
                validated
            )
        }
    }
}

// MARK: - Helpers

private struct SendableSecKey: @unchecked Sendable {
    let secKey: SecKey
}

// MARK: - Errors

extension EnsembleHPKE {
    public enum Error: Swift.Error {
        case expiredAttestation
        case unsupportedKeyAlgorithm
    }
}

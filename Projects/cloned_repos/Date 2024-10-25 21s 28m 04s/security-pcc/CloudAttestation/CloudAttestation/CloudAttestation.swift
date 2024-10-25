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
//  CloudAttestation.swift
//  CloudAttestation
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import Foundation
import Security
import CryptoKit
import os.log
import DarwinPrivate.os.variant

// MARK: - Attestor Protocol

/// Attestor defines a protocol that can attest the presence of a `SecKey`.
///
/// Attestors also implement ``Validator`` and must be thread safe.
public protocol Attestor: Sendable {
    var defaultKeyLifetime: TimeInterval { get }
    /// Attest that a key is resident on the current device's Secure Enclave, along with all relevant attestable data.
    ///
    /// Attestation from a Secure Enclave must be performed against a SEP resident key, even if only the attestation structures about the device state are desired.
    ///
    /// - Parameters:
    ///  - key: A SecKey that will be attested by DCIK.
    ///  - expiration: the expiration date to staple in the AttestationBundle. Defaults to ``Attestor/defaultKeyDuration`` (24 hours)
    ///  - nonce: An optional nonce value that can be up to 32 bytes. Function will throw if larger.
    func attest(key: SecKey, expiration: Date, nonce: Data?) async throws -> AttestationBundle
}

extension Attestor {

    public var defaultKeyLifetime: TimeInterval {
        TimeInterval(defaultKeyDuration.components.seconds)
    }

    /// The default key lifetime, returned as a Duration object.
    public var defaultKeyDuration: Swift.Duration {
        .days(1)
    }

    /// Helper method that will call ``Attestor/attest(key:expiration:policy:)`` with ``Validator/defaultPolicy-swift.property`` and a default expiration of ``Attestor/defaultKeyLifetime`` from now.
    @inlinable
    public func attest(key: SecKey) async throws -> AttestationBundle {
        try await self.attest(key: key, expiration: Date(timeIntervalSinceNow: self.defaultKeyLifetime), nonce: nil)
    }

    /// Helper method that will call ``Attestor/attest(key:expiration:nonce:)`` with an empty nonce.
    @inlinable
    public func attest(key: SecKey, expiration: Date) async throws -> AttestationBundle {
        try await self.attest(key: key, expiration: expiration, nonce: nil)
    }

    /// Returns a default ``AttestationBundle`` that can be used for testing.
    /// - Parameters:
    ///   - key: The ``SecKey`` to attest.
    ///   - nonce: The nonce to use for testing.
    @inlinable
    public func attest(key: SecKey, nonce: Data?) async throws -> AttestationBundle {
        try await self.attest(key: key, expiration: Date(timeIntervalSinceNow: self.defaultKeyLifetime), nonce: nonce)
    }
}

// MARK: - Validator Protocol

/// Validator defines a protocol that can validate ``AttestationBundle`` returned from ``Attestor``.
public protocol Validator: Sendable {
    associatedtype DefaultPolicy: AttestationPolicy

    /// Computes a default ``AttestationPolicy`` assosciated with this type.
    var defaultPolicy: DefaultPolicy { get }

    /// Validates an attestation bundle with a provided policy.
    ///
    /// Validation of an AttestationBundle will evaluate the provided ``AttestationPolicy`` against the the provided ``AttestationBundle``. Validation may be done asynchronously.
    /// Regardless of whether the Policy is empty or not, this method will always at minimum, parse the provided ``SEP/Attestation`` and extract the ``PublicKeyData``
    ///
    /// - Parameters:
    ///     - bundle: the ``AttestationBundle`` to evaluate.
    ///     - policy: some ``AttestationPolicy`` to evaluate the provided bundle against.
    ///
    /// - Returns: A tuple containing the Byte encoded form of the Public Key, an expiration Date, and a validated copy of the ``Validated/AttestationBundle``
    ///
    /// - Throws: ``CloudAttestationError`` if the attestation is invalid, or a policy rejected it.
    func validate(
        bundle: AttestationBundle,
        nonce: Data?,
        policy: some AttestationPolicy
    ) async throws -> (key: PublicKeyData, expiration: Date, attestation: Validated.AttestationBundle)
}

extension Validator {
    /// Default validate implementation
    public func validate(
        bundle: AttestationBundle,
        nonce: Data?,
        policy: some AttestationPolicy
    ) async throws -> (key: PublicKeyData, expiration: Date, attestation: Validated.AttestationBundle) {
        let logger = Logger(subsystem: "com.apple.CloudAttestation", category: String(describing: Self.self))
        do {
            var context = AttestationPolicyContext()
            try await policy.evaluate(bundle: bundle, context: &context)

            let attestation: SEP.Attestation
            if let maybeAttestation = context[SEPAttestationPolicy.validatedAttestationKey] as? SEP.Attestation {
                attestation = maybeAttestation
            } else {
                attestation = try SEP.Attestation(from: bundle.proto.sepAttestation)
            }

            if let nonce {
                guard attestation.nonce == nonce else {
                    throw CloudAttestationError.invalidNonce
                }
            }

            guard let keyData: PublicKeyData = attestation.publicKeyData else {
                throw CloudAttestationError.unexpected(reason: "Unknown public key type")
            }

            logger.log("AttestationBundle passed validation for public key: \(keyData.fingerprint())")

            let expiration = bundle.proto.keyExpiration.date
            guard Date.now < expiration else {
                throw CloudAttestationError.expired(expiration: expiration)
            }
            return (
                key: keyData, expiration: expiration,
                attestation: Validated.AttestationBundle(bundle: bundle, udid: attestation.identity?.udid, routingHint: nil)
            )
        } catch {
            logger.error("AttestationBundle validation failed: \(error)")
            throw error
        }
    }

    /// Validates a bundle and returns a tuple of the key data, the expiration date, and the attestation bundle.
    /// - Parameter bundle: The attestation bundle to validate.
    @inlinable
    public func validate(bundle: AttestationBundle) async throws -> (key: PublicKeyData, expiration: Date, attestation: Validated.AttestationBundle) {
        try await self.validate(bundle: bundle, nonce: nil, policy: self.defaultPolicy)
    }

    /// Validates a bundle with an optional nonce and returns a tuple of the key data, the expiration date, and the attestation bundle.
    /// - Parameters:
    ///   - bundle: The attestation bundle to validate.
    ///   - nonce: The nonce to validate.
    @inlinable
    public func validate(bundle: AttestationBundle, nonce: Data?) async throws -> (key: PublicKeyData, expiration: Date, attestation: Validated.AttestationBundle) {
        try await self.validate(bundle: bundle, nonce: nonce, policy: self.defaultPolicy)
    }

    /// Validates a bundle with a custom attestation policy and returns a tuple of the key data, the expiration date, and the attestation bundle.
    /// - Parameters:
    ///   - bundle: The attestation bundle to validate.
    ///   - policy: The policy to validate the bundle against.
    @inlinable
    public func validate(bundle: AttestationBundle, policy: some AttestationPolicy) async throws -> (key: PublicKeyData, expiration: Date, attestation: Validated.AttestationBundle)
    {
        try await self.validate(bundle: bundle, nonce: nil, policy: policy)
    }

    /// Validates a bundle with a custom policy built with ``PolicyBuilder`` and returns a tuple of the key data, the expiration date, and the attestation bundle.
    /// - Parameters:
    ///   - bundle: The attestation bundle to validate.
    ///   - nonce: The nonce to validate.
    ///   - _: The nonce to validate.
    @inlinable
    public func validate(
        bundle: AttestationBundle,
        nonce: Data?,
        @PolicyBuilder _ builder: () -> some AttestationPolicy
    ) async throws -> (key: PublicKeyData, expiration: Date, attestation: Validated.AttestationBundle) {
        try await self.validate(bundle: bundle, nonce: nonce, policy: builder())
    }

    /// Validates a bundle with a custom policy built with ``PolicyBuilder`` and returns a tuple of the key data, the expiration date, and the attestation bundle.
    /// - Parameters:
    ///   - bundle: The attestation bundle to validate.
    ///   - _: The nonce to validate.
    @inlinable
    public func validate(
        bundle: AttestationBundle,
        @PolicyBuilder _ builder: () -> some AttestationPolicy
    ) async throws -> (key: PublicKeyData, expiration: Date, attestation: Validated.AttestationBundle) {
        try await self.validate(bundle: bundle, nonce: nil, policy: builder())
    }
}

// MARK: - Attestation Bundle

/// Encapsulates an AttestationBundle parsed from its wire format.
public struct AttestationBundle: Equatable, Sendable {
    static let logger = Logger(subsystem: "com.apple.CloudAttestation", category: "AttestationBundle")

    var proto: Proto_AttestationBundle

    init(proto: Proto_AttestationBundle) {
        self.proto = proto
    }

    /// Construct an ``AttestationBundle`` from the Data bytes of a protobuf serialized AttestationBundle
    public init(data: Data) throws {
        self.proto = try Proto_AttestationBundle(serializedBytes: data)
    }

    /// Construct an ``AttestationBundle`` from the protobuf-json serialized string of an AttestationBundle
    public init(jsonString json: String) throws {
        self.proto = try Proto_AttestationBundle(jsonString: json)
    }

    /// Accepts a closure allowing for the access of certain attestation bundle fields without having done any validation.
    /// - Parameter body: The attestation bundle to validate.
    public func withUnvalidatedAttestationBundle<R>(_ body: (Unvalidated.AttestationBundle) throws -> R) rethrows -> R {
        try body(Unvalidated.AttestationBundle(bundle: self))
    }
}

extension AttestationBundle: Encodable {
    /// Performs the encoding of the attestation bundle.
    /// - Parameter encoder: The encoder to encode the attestation bundle.
    public func encode(to encoder: any Encoder) throws {
        try self.proto.encode(to: encoder)
    }
}

public enum Unvalidated {
    public struct AttestationBundle: Equatable, Sendable {
        static let logger = Logger(subsystem: "com.apple.CloudAttestation", category: "Unvalidated.AttestationBundle")

        let bundle: CloudAttestation.AttestationBundle

        /// The UDID of the node that the attestation was generated for
        public var udid: String? {
            do {
                let attestation = try SEP.Attestation(from: bundle.proto.sepAttestation)
                guard let udid = attestation.identity?.udid else {
                    Self.logger.error("Unable to parse device udid from sep attestation")
                    return nil
                }
                Self.logger.log("Parsed udid=\(udid, privacy: .public) from unvalidated attestation bundle")
                return udid
            } catch {
                Self.logger.error("Invalid sep attestation blob: \(error, privacy: .public)")
                return nil
            }
        }
    }
}

/// Encaspulates validated structures
public enum Validated {
    /// Exposes a validated AttestationBundle that allows access to fields.
    public struct AttestationBundle: Equatable, Sendable {
        static let logger = Logger(subsystem: "com.apple.CloudAttestation", category: "Validated.AttestationBundle")

        @_spi(Private)
        public let bundle: CloudAttestation.AttestationBundle

        /// The raw SEP attestation blob.
        public var sepAttestation: Data { bundle.proto.sepAttestation }

        let _udid: String?
        /// The UDID of the node that the attestation was generated for.
        public var udid: String? {
            if let _udid {
                return _udid
            }
            Self.logger.warning("UDID was not set from validation context, attempting to lazily parse from sep attestation blob")
            do {
                let attestation = try SEP.Attestation(from: sepAttestation)
                guard let udid = attestation.identity?.udid else {
                    Self.logger.error("Unable to parse device udid from sep attestation")
                    return nil
                }
                return udid
            } catch {
                Self.logger.error("Unexpected invalid sep attestation blob (how did this happen?): \(error, privacy: .public)")
                return nil
            }
        }

        /// The public key used to sign the attestation.
        public var provisioningCertificateChain: [Data] { bundle.proto.provisioningCertificateChain }

        /// The public key used to sign the attestation.
        public var keyExpiration: Date { bundle.proto.keyExpiration.date }

        /// The routing hint used to route the attestation to the correct device.
        public let routingHint: String?

        @available(*, deprecated)
        public let ensembleUDIDs: [String]? = nil

        init(bundle: CloudAttestation.AttestationBundle, udid: String?, routingHint: String?) {
            self.bundle = bundle
            self._udid = udid
            self.routingHint = routingHint
        }
    }
}

/// Enumerates the different types of byte encoded Public Key formats.
public enum PublicKeyData: Sendable, Equatable {
    /// ANSI X9.63 encoded ECC Public Key
    case x963(Data)
    /// Curve25519 encoded Public Key
    case curve25519(Data)

    func fingerprint<Hash: HashFunction>(using: Hash.Type = SHA256.self) -> Hash.Digest {
        switch self {
        case .x963(let data):
            fallthrough

        case .curve25519(let data):
            return Hash.hash(data: data)
        }
    }

    public var raw: Data {
        switch self {
        case .x963(let data):
            fallthrough
        case .curve25519(let data):
            return data
        }
    }
}

// MARK: - Serialization

extension AttestationBundle {
    /// Returns the bundle as protobuf encoded bytes.
    public func serializedData() throws -> Data {
        return try self.proto.serializedData()
    }

    /// Returns the bundle as a json encoded string.
    public func jsonString() throws -> String {
        return try self.proto.jsonString()
    }
}

// MARK: - Debugging

extension AttestationBundle {
    /// Writes the attestation bundle to a directory for tracing
    func trace() {
        guard os_variant_allows_internal_security_policies("com.apple.CloudAttestation") else {
            return
        }

        let traceDirectory = CFPreferencesCopyAppValue("traceDirectory" as CFString, "com.apple.CloudAttestation" as CFString) as? String

        guard let traceDirectory else {
            return
        }

        do {
            let data = try self.serializedData()

            let directory = URL(fileURLWithPath: traceDirectory).appending(path: Environment.current.description, directoryHint: .isDirectory)
            let file = directory.appending(component: "\(SHA256.hash(data: data).hexString).attestation")

            Self.logger.debug("Writing attestation bundle to \(file, privacy: .public)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try data.write(to: file, options: .withoutOverwriting)
        } catch {
            Self.logger.error("Tracing attestation bundle failed with error \(error, privacy: .public)")
        }
    }
}

// MARK: - Error API

public enum CloudAttestationError: Error {
    case unexpected(reason: String)
    @available(*, deprecated) case attestError(Error)
    @available(*, deprecated) case validateError(Error)
    case invalidNonce
    case expired(expiration: Date)
}

extension CloudAttestationError: CustomNSError {
    public static var errorDomain: String = "com.apple.CloudAttestation.CloudAttestationError"

    public var errorCode: Int {
        switch self {
        case .unexpected(_): return 1
        case .attestError(_): return 2
        case .validateError(_): return 3
        case .invalidNonce: return 4
        case .expired(_): return 5
        }
    }
}

#if DEBUG
// enforce compiler error if this is still present in release builds
extension CloudAttestationError {
    static var notImplemented: Self {
        .unexpected(reason: "Not implemented")
    }
}
#endif

// MARK: - Transparency Auditor API

extension AttestationBundle {
    @_spi(TransparencyAuditor)
    public var atLogProofs: Data {
        get throws {
            try self.proto.transparencyProofs.proofs.serializedData()
        }
    }
}

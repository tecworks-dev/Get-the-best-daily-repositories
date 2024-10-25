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

import CloudAttestation
import CloudBoardAttestationDAPI
import CloudBoardCommon
import CloudBoardMetrics
@_spi(SEP_Curve25519) import CryptoKit
@_spi(SEP_Curve25519) import CryptoKitPrivate
import Foundation
import os
import Security_Private.SecItemPriv
import Security_Private.SecKeyPriv

enum CloudAttestationProviderError: Error {
    case sepUnavailable
    case failedToAccessControlFlags(Error)
    case failedToDeleteExistingKey(Error)
    case failedToCreateKey(Error)
    case failedToObtainPersistentKeyReference(OSStatus)
    case failedToParseTransparencyURL(String)
    case earlyExit
    case cloudAttestationUnavailable
}

/// Provides attested key with an attestation provided by CloudAttestation.framework
struct CloudAttestationProvider: AttestationProvider {
    fileprivate static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "CloudAttestationProvider"
    )

    private var configuration: CloudBoardAttestationDConfiguration
    private var keychain: SecKeychain?
    private var metrics: MetricsSystem

    init(configuration: CloudBoardAttestationDConfiguration, keychain: SecKeychain? = nil, metrics: MetricsSystem) {
        self.configuration = configuration
        self.keychain = keychain
        self.metrics = metrics
    }

    func createAttestedKey(attestationBundleExpiry: Date) async throws -> InternalAttestedKey {
        Self.logger.info("Creating attested SEP-backed X25519 key")

        guard SecureEnclave.isAvailable else {
            Self.logger.fault("SEP is unavailable. Cannot create an attested SEP-backed key.")
            fatalError("SEP is unavailable. Cannot create an attested SEP-backed key.")
        }

        let retryConfig = self.configuration.cloudAttestation.attestationRetryConfiguration
        let retryStrategy = self.createCloudAttestationRetryStrategy(retryConfig: retryConfig)

        return try await self.metrics.withStatusMetrics(
            total: Metrics.CloudAttestation.AttestationCounter(action: .increment(by: 1)),
            error: Metrics.CloudAttestation.AttestationErrorCounter.Factory()
        ) {
            return try await executeWithRetries(
                retryStrategy: retryStrategy,
                perRetryTimeout: retryConfig.perRetryTimeout
            ) {
                let privateKey = try createSecureEnclaveKey()
                let publicKey = try SecureEnclave.Curve25519.KeyAgreement.PrivateKey(from: privateKey).publicKey
                let attestationBundle = try await createAttestationBundle(
                    for: privateKey,
                    expiry: attestationBundleExpiry
                )

                Self.logger.notice(
                    "Created attested SEP-backed key with public key \(publicKey.rawRepresentation.base64EncodedString(), privacy: .public)"
                )

                return InternalAttestedKey(key: .sepKey(secKey: privateKey), attestationBundle: attestationBundle)
            }
        }
    }

    private func createSecureEnclaveKey() throws -> SecKey {
        var error: Unmanaged<CFError>?
        guard let aclOpts = SecAccessControlCreateWithFlags(
            kCFAllocatorDefault,
            kSecAttrAccessibleAlwaysThisDeviceOnlyPrivate,
            .privateKeyUsage,
            &error
        ) else {
            // If this returns nil, error must be set
            throw CloudAttestationProviderError.failedToAccessControlFlags(error!.takeRetainedValue() as Error)
        }

        // Create key
        var attributes = Keychain.baseNodeKeyQuery
        attributes[kSecAttrIsPermanent as String] = false
        attributes.addKeychainAttributes(keychain: self.keychain, for: .update)
        attributes[kSecPrivateKeyAttrs as String] = [
            kSecAttrAccessControl: aclOpts,
            kSecAttrLabel: "CloudBoard X25519 Key",
            kSecKeyOSBound: true,
            kSecKeySealedHashesBound: true,
        ] as [String: Any]
        guard let secKey = SecKeyCreateRandomKey(attributes as CFDictionary, &error) else {
            // If this returns nil, error must be set
            throw CloudAttestationProviderError.failedToCreateKey(error!.takeRetainedValue() as Error)
        }

        return secKey
    }

    private func createCloudAttestationRetryStrategy(
        retryConfig: CloudBoardAttestationDConfiguration
            .RetryConfiguration
    ) -> RetryStrategy {
        return RetryWithBackoff(
            backoffStrategy: ExponentialBackoffStrategy(from: retryConfig),
            deadline: retryConfig.timeout.map { .instant(.now + $0) } ?? .noDeadline,
            retryFilter: { error in
                // We defensively retry on all errors as we don't have an exhaustive set of known retryable errors and
                // the set might change over time.
                self.metrics.emit(Metrics.CloudAttestation.AttestationRetryCounter(action: .increment(by: 1)))
                self.metrics.emit(Metrics.CloudAttestation.AttestationErrorCounter.Factory().make(error))
                Self.logger.error(
                    "Failed to generate attestation with error: \(String(unredacted: error), privacy: .public). Retrying with backoff."
                )
                return .continue
            }
        )
    }

    private func createAttestationBundle(for privateKey: SecKey, expiry: Date) async throws -> Data {
        guard self.configuration.cloudAttestation.enabled else {
            Self.logger.warning("Attestation via CloudAttestation.framework disabled. Using fake attestation bundle.")
            let publicKey = try SecureEnclave.Curve25519.KeyAgreement.PrivateKey(from: privateKey).publicKey
            return try FakeAttestationBundle.data(for: publicKey, kem: .Curve25519_HKDF_SHA256)
        }

        guard #_hasSymbol(NodeAttestor.self) else {
            Self.logger.fault(
                "CloudAttestation.framework unavailable. Unable to generate CloudAttestation attestation bundle."
            )
            throw CloudAttestationProviderError.cloudAttestationUnavailable
        }

        let attestor: NodeAttestor
        if self.configuration.cloudAttestation.includeTransparencyLogInclusionProof {
            Self.logger.log("CloudAttestation transparency proof inclusion enabled")
            attestor = NodeAttestor()
        } else {
            Self.logger.warning("CloudAttestation transparency proof inclusion disabled")
            attestor = NodeAttestor(transparencyProver: NopTransparencyLog())
        }

        let attestationBundle = try await attestor.attest(key: privateKey, expiration: expiry)
        let attestationBundleJson = try attestationBundle.jsonString()
        Self.logger.debug("Generated key attestation bundle: \(attestationBundleJson, privacy: .public)")

        return try attestationBundle.serializedData()
    }
}

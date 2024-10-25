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
//  EnsembleValidator.swift
//  CloudAttestation
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import IOKit
import FeatureFlags
@preconcurrency import Security
import Security_Private.SecKeyPriv
import os.log
@_weakLinked import SecureConfigDB

public struct EnsembleValidator: Validator {
    static let logger = Logger(subsystem: "com.apple.CloudAttestation", category: "EnsembleValidator")

    var release: Release
    var chipIdentity: SEP.Identity
    var boardID: UInt32
    var hardwareIdentifiers: HardwareIdentifiersPolicy.Identifiers {
        (archBits: chipIdentity.archBits, chipID: chipIdentity.chipID, boardID: boardID)
    }
    var restrictedExecution: DeviceModePolicy.Constraint
    var ephemeralData: DeviceModePolicy.Constraint
    var developer: DeviceModePolicy.Constraint
    var cryptexLockdown: Bool
    var securityPolicy: DarwinInit.SecureConfigSecurityPolicy

    static let dcik = SecKeyCopySystemKey(.DCIK, nil)?.takeRetainedValue()

    // Defang parameters
    @_spi(Private)
    public var roots: [SecCertificate]

    @_spi(Private)
    public var clock: Date?

    @_spi(Private)
    public var strictCertificateValidation: Bool

    @_spi(Private)
    public var requireProdTrustAnchors: Bool

    @_spi(Private)
    public var checkRevocation: Bool

    /// Creates a new ``EnsembleValidator``.
    public init() throws {
        self = try .init(assetProvider: DefaultAssetProvider())
    }

    @_spi(Private)
    public init(assetProvider: some AttestationAssetProvider) throws {
        self.roots = []
        self.clock = nil
        (self.chipIdentity, self.boardID) = try Self.localChipIdentity()
        self.release = try Release.local(assetProvider: assetProvider)
        let (mode, cryptexLockdown) = try Self.localDeviceStates()
        self.restrictedExecution = .init(mode.restrictedExecution)
        self.ephemeralData = .init(mode.ephemeralData)
        self.developer = .init(mode.developer)
        self.cryptexLockdown = cryptexLockdown

        let provisioningCertChain = try? assetProvider.provisioningCertificateChain

        // if we have a cert, we require our peers to have a cert
        self.strictCertificateValidation = provisioningCertChain?.isEmpty == false
        if let provisioningCertChain {
            // if we have a production cert, require our peers to have a production cert
            self.requireProdTrustAnchors = Self.isProductionCert(chain: provisioningCertChain)
            self.checkRevocation = true
        } else {
            self.requireProdTrustAnchors = false
            self.checkRevocation = false
        }

        if #_hasSymbol(SecureConfigParameters.self) {
            let secureConfig = try SecureConfigParameters.loadContents()
            switch secureConfig.securityPolicy {
            case .none:
                self.securityPolicy = .none
            case .carry:
                self.securityPolicy = .carry
            case .customer:
                self.securityPolicy = .customer
            @unknown default:
                Self.logger.warning("Unknown config security policy \(secureConfig.securityPolicy.rawValue, privacy: .public), defaulting to customer")
                self.securityPolicy = .customer
            }
        } else {
            // This is really only for Unit Tests
            self.securityPolicy = .customer
        }
    }

    static func isProductionCert(chain: [Data]) -> Bool {
        let policy = X509Policy(roots: X509Policy.prodProvisioningRoots)
        return (try? policy.evaluateCertificateChain(chain)) != nil
    }

    /// The default policy.
    public var defaultPolicy: some AttestationPolicy {
        self.buildPolicy(for: nil, fingerprints: nil)
    }

    /// Returns a policy for the given UDID.
    /// - Parameter udid: The UDID.
    public func policyFor(udid: String) -> some AttestationPolicy {
        self.buildPolicy(for: udid, fingerprints: nil)
    }

    /// Returns a policy for the given UDID and ensemble certificate fingerprints.
    /// - Parameters:
    ///   - udid: The UDID.
    ///   - fingerprints: The fingerprints.
    public func policyFor(udid: String, fingerprints: [Data]) -> some AttestationPolicy {
        self.buildPolicy(for: udid, fingerprints: fingerprints)
    }

    /// Returns a policy for the given SEP identity.
    /// - Parameter identity: The identity.
    public func policyFor(identity: SEP.Identity) -> some AttestationPolicy {
        self.buildPolicy(for: identity.udid, fingerprints: nil)
    }

    var trustAnchors: [SecCertificate] {
        if requireProdTrustAnchors {
            X509Policy.prodProvisioningRoots
        } else {
            self.roots + X509Policy.prodProvisioningRoots + X509Policy.testProvisioningRoots
        }
    }

    @PolicyBuilder
    private func buildPolicy(for udid: String?, fingerprints: [Data]?) -> some AttestationPolicy {
        let revocationPolicy: X509Policy.RevocationPolicy? = self.checkRevocation ? [.any] : nil
        X509Policy(required: self.strictCertificateValidation, roots: self.trustAnchors, clock: self.clock, revocation: revocationPolicy)
        if let fingerprints {
            X509FingerprintPolicy(fingerprints: fingerprints)
        }
        SEPAttestationPolicy(insecure: !self.strictCertificateValidation).verifies { attestation in
            if let udid = udid {
                let udidsMatch = attestation.identity?.udid == udid
                let _ =
                    udidsMatch
                    ? (Self.logger.log("Attestation udid \(attestation.identity?.udid ?? "<unknown>", privacy: .public) matches \(udid, privacy: .public)"))
                    : Self.logger.error("Attestation udid \(attestation.identity?.udid ?? "<unknown>", privacy: .public) does not match \(udid, privacy: .public)")
                udidsMatch
            }
        }
        APTicketPolicy()
        SEPImagePolicy()
        CryptexPolicy(locked: self.cryptexLockdown)
        SecureConfigPolicy()
        KeyOptionsPolicy(mustContain: [.osBound, .sealedHashesBound])
        SoftwareReleasePolicy(release: self.release)
        // SuperSet of FusingPolicy
        HardwareIdentifiersPolicy(matches: self.hardwareIdentifiers)
        DeviceModePolicy(
            restrictedExecution: self.restrictedExecution,
            ephemeralData: self.ephemeralData,
            developer: self.developer
        )
        DarwinInitPolicy(securityPolicy: self.securityPolicy)
    }
}

extension EnsembleValidator {
    static func localDeviceStates() throws -> (deviceMode: DeviceModePolicy.Mode, cryptedLockdown: Bool) {
        guard let dcik else {
            throw Error.missingDCIK
        }
        guard let dcikPub = SecKeyCopyPublicKey(dcik) else {
            throw Error.missingDCIK
        }
        let dummyKey = try EnsembleHPKE.createSEPKey()
        var error: Unmanaged<CFError>?
        guard let attestData = SecKeyCreateAttestation(dcik, dummyKey, &error)?.takeRetainedValue() as? Data else {
            throw Error.introspectionError(.attestationError(underlying: error!.takeRetainedValue()))
        }
        let attest = try SEP.Attestation(from: attestData, signer: dcikPub)
        let cryptexLockdown = attest.sealedHash(at: cryptexSlotUUID)?.flags.contains(.ratchetLocked) ?? false
        return (
            DeviceModePolicy.Mode(
                restrictedExecution: attest.restrictedExecutionMode ?? false,
                ephemeralData: attest.ephemeralDataMode ?? false,
                developer: attest.developerMode ?? true
            ), cryptexLockdown
        )
    }

    static func localChipIdentity() throws -> (identity: SEP.Identity, boardID: UInt32) {
        guard let entry = IORegistryEntry(path: "IODeviceTree:/chosen") else {
            throw Error.introspectionError(.missingIODeviceTreeChosen)
        }

        guard let SDOM: UInt32 = entry["security-domain"], let securityDomain = SEP.Identity.ArchBits.SecurityDomain(rawValue: UInt8(SDOM)) else {
            throw Error.introspectionError(.missingSecurityDomain)
        }

        guard let ESEC: UInt32 = entry["effective-security-mode-ap"] else {
            throw Error.introspectionError(.missingSecurityMode)
        }

        guard let EPRO: UInt32 = entry["effective-production-status-ap"] else {
            throw Error.introspectionError(.missingProductionStatus)
        }

        guard let ECID: UInt64 = entry["unique-chip-id"] else {
            throw Error.introspectionError(.missingECID)
        }

        guard let CHIP: UInt32 = entry["chip-id"] else {
            throw Error.introspectionError(.missingChipID)
        }

        guard let BORD: UInt32 = entry["board-id"] else {
            throw Error.introspectionError(.missingBoardID)
        }

        return (
            // SW_SEED is not exposed in IORegistry, so just set to 0.
            identity: SEP.Identity(chipID: CHIP, ecid: ECID, archBits: .init(productionStatus: EPRO == 1, securityMode: ESEC == 1, securityDomain: securityDomain), swSeed: 0),
            boardID: BORD
        )
    }
}

// MARK: - EnsembleValidator Error API

extension EnsembleValidator {
    public enum Error: Swift.Error {
        case introspectionError(IntrospectionError)
        case missingDCIK
    }
}

extension EnsembleValidator.Error {
    public enum IntrospectionError: Swift.Error {
        case missingIODeviceTreeChosen
        case missingSecurityDomain
        case missingSecurityMode
        case missingProductionStatus
        case missingBoardID
        case missingChipID
        case missingECID
        case missingAPTicket
        case missingSealedHashEntries
        case missingCryptexSlot
        case missingDarwinInit
        case attestationError(underlying: Swift.Error)
    }
}

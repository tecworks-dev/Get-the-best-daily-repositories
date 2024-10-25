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
//  NarrativeHelper.swift
//  CloudMetricsDaemon
//
//  Created by Dhanasekar Thangavel on 1/10/23.
//

import CloudMetricsFramework
import NIOCore
import NIOSSL
import os
import libnarrativecert
import MobileGestaltPrivate
import notify

// swiftlint:disable file_length
private let logger = Logger(subsystem: kCloudMetricsLoggingSubsystem, category: "NarrativeHelper")

internal enum NarrativeHelperError: Error, CustomStringConvertible {
    case unableToGetCertChain(String)
    case noUserGroupOrNamespaceGroup
    case unableToGetCertificate(String)

    internal var description: String {
        switch self {
        case .unableToGetCertChain(let error):
            return "Unable to get certificate chain: \(error)"
        case .noUserGroupOrNamespaceGroup:
            return "Error reading UserGroup and NamespaceGroup"
        case .unableToGetCertificate(let error):
            return "Unable to get certificate : \(error)"
        }
    }
}

internal enum NarrativeSignError: LocalizedError {
    case invalidSignature
    case signingFailed(Error)
    case unsupportedAlgorithm(SignatureAlgorithm)

    internal var errorDescription: String? {
        switch self {
        case .invalidSignature:
            return "invalid signature"
        case .signingFailed(let error):
            return "unable to sign: \(error)"
        case .unsupportedAlgorithm(let algorithm):
            return String(format: "unable to sign using algorithm: 0x%04x", algorithm.rawValue)
        }
    }
}

// callback function signature for certificate renewal callback that client can register to.
internal typealias RenewCertificateCallback = (_ newCert: CloudMetricsCertConfig ) async throws -> Void
internal typealias CertExpiryCallback = () async -> Void
internal typealias CertExpiryContinuation = AsyncStream<CertExpiryCallback>.Continuation

internal class NICCertExpiryHandler {
    private let serialQueue: DispatchQueue
    private let renewCallback: RenewCertificateCallback
    private let logger: Logger
    private let clientStream: AsyncStream<CertExpiryCallback>
    private let clientStreamContinuation: AsyncStream<CertExpiryCallback>.Continuation
    private var token : Int32 = 0
    
    internal init(
        renewCallback: @escaping RenewCertificateCallback
    ) throws {
        self.serialQueue = DispatchQueue(label: "com.apple.acdc.cloudmetricsd.narrativedispatch", qos: .userInitiated)
        self.renewCallback = renewCallback
        self.logger = Logger(subsystem: kCloudMetricsLoggingSubsystem, category: "NarrativeHelper")
        // swiftlint:disable:next implicitly_unwrapped_optional
        var continuation: AsyncStream<CertExpiryCallback>.Continuation! = nil
        self.clientStream = AsyncStream<CertExpiryCallback> { continuation = $0 }
        self.clientStreamContinuation = continuation

        Task {
            for await callback in clientStream {
                await callback()
            }
        }
    }

    // registers the cert renewal notification callback with libnarrativecert
    internal func registerCertRenewalNotification() {
        
        let acdcActorCert = NarrativeCert(
            domain: .acdc,
            identityType: .actor
        )
       // TODO: replace with acdcActorCert.refreshedNotificationName once the changes are in laetst OS SDK
        notify_register_dispatch("com.apple.narrativecertd.acdc.actor.notification.refreshed", &token, DispatchQueue.global(qos: .utility)) { _ in
            self.logger.debug("Got refreshed notification")
            self.clientStreamContinuation.yield {
                do {
                    let acdcActorCert = NarrativeCert(
                        domain: .acdc,
                        identityType: .actor
                    )
                   
                    let tlsCertConfig = try await getCloudMetricsCertConfigFromNarrativeIdentity(narrativeCert: acdcActorCert)
                    try await self.renewCallback(tlsCertConfig)
                    
                    notify_cancel(self.token)
                    
                    self.registerCertRenewalNotification()
                } catch {
                    self.logger.error("Not able to renew the cert error:\(error, privacy: .public)")
                }
            }
        }
    }

    deinit {
        clientStreamContinuation.finish()
    }
}

internal func getCertificateName(_ certificate: SecCertificate) -> String {
    var commonName: CFString? = nil
    SecCertificateCopyCommonName(certificate, &commonName)

    if let name = commonName {
        return name as String
    }
    return "<missing common name>"
}

//TODO: Remove this
public func getFullCertChain(narrativeCert: NarrativeCert)  -> [SecCertificate]? {
    let narrativeRef = narrativeCert.fetchSecRefsFromKeychain()
    guard let narrativeRef = narrativeRef else {
        return nil
    }
    
    var certChain = [narrativeRef.certRef]
    let rootChain = narrativeCert.getCertChain()
    for crt in rootChain {
        guard let crtData = Data(base64Encoded: crt) else {
            logger.error("unable to decode certificate obtained from rootchain.")
            return nil
        }
        guard let cert = SecCertificateCreateWithData(kCFAllocatorDefault,  crtData as CFData) else {
            logger.error("unable to create certificate from data.")
            return nil
        }
        certChain.append(cert)
    }
    
    return certChain
}


// gets the CloudMetricsCertConfig from NarrativeIdentity
internal
func getCloudMetricsCertConfigFromNarrativeIdentity(narrativeCert: NarrativeCert) async throws -> CloudMetricsCertConfig {
    let narrativeRef = narrativeCert.fetchSecRefsFromKeychain()
    
    logger.debug("Getting sec references")
    guard let narrativeRef = narrativeRef else {
        throw NarrativeHelperError.unableToGetCertificate("Unable to get Sec references")
    }
    
    logger.debug("creating NIOSSLPrivatekey")
    let customKey =  NarrativeKey(key: narrativeRef.keyRef)
    let mtlsPrivateKey = NIOSSLPrivateKey(customPrivateKey: customKey)

    logger.debug("getting cert chain")
    // TODO: To uncomment this when on latest SDK
    // let certChain = narrativeCert.getFullCertChain()
    let fullCertChain = getFullCertChain(narrativeCert: narrativeCert)
    guard let certChain = fullCertChain else {
        let errorMessage = "Unable to obtain certChain from narrative Identity"
        throw NarrativeHelperError.unableToGetCertChain(errorMessage)
    }

    logger.debug("mapping cert chain")
    let mtlsCertificateChain = try certChain.map {
        try NIOSSLCertificate(bytes: [UInt8](SecCertificateCopyData($0) as NSData as Data), format: .der)
    }

    let cryptexMountPoint = ProcessInfo.processInfo.environment["CRYPTEX_MOUNT_PATH"] ?? ""

    var trustRootsRelativePath = "/usr/share/cloudmetricsd/mosaic_trustroot.pem"
#if os(iOS)
		if MobileGestalt.current.isComputeController {
			trustRootsRelativePath = "/usr/share/cloudmetricsd_bmc/mosaic_trustroot.pem"
		}
#endif
    let trustRootsFilePath = "\(cryptexMountPoint)\(trustRootsRelativePath)"
    let mtlsTrustRoots = NIOSSLTrustRoots.certificates([
        try NIOSSLCertificate(file: trustRootsFilePath, format: .pem),
    ])

    let hostCert = NarrativeCert(domain: .adb, identityType: .host)
    guard let hostCertRefs = hostCert.fetchSecRefsFromKeychain() else {
        throw NarrativeHelperError.unableToGetCertificate("Error getting SecRefs for adb host certificate")
    }
    
    let certName  = getCertificateName(hostCertRefs.certRef)
    logger.debug("""
                      Narrative host cert details
                      Identifier=\(hostCert.keychainLabel), \
                      CommonName=\(certName)
                """)

    return CloudMetricsCertConfig(mtlsPrivateKey: mtlsPrivateKey,
                                  mtlsCertificateChain: mtlsCertificateChain,
                                  mtlsTrustRoots: mtlsTrustRoots,
                                  hostName: certName)
}

internal struct NarrativeKey: NIOSSLCustomPrivateKey, Hashable {
    private static let queue = DispatchQueue(label: "com.apple.acdc.cloudmetrics.narrative-tls-key-signing")

    internal var signatureAlgorithms: [SignatureAlgorithm]
    private var key: SecKey

    internal init(key: SecKey, signatureAlgorithms: [SignatureAlgorithm] = [.ecdsaSecp256R1Sha256]) {
        self.signatureAlgorithms = signatureAlgorithms
        self.key = key
    }

    internal func sign(
        channel: Channel,
        algorithm: SignatureAlgorithm,
        data buffer: ByteBuffer
    ) -> EventLoopFuture<ByteBuffer> {
        // make a promise for the channel's event loop
        let promise = channel.eventLoop.makePromise(of: ByteBuffer.self)

        // on the dispatch queue for this narrative key, async the signing
        // process with SecKeyCreateSignature (which is blocking) and succeed or
        // fail the promise in the async block
        Self.queue.async {
            do {
                guard case .ecdsaSecp256R1Sha256 = algorithm else {
                    throw NarrativeSignError.unsupportedAlgorithm(algorithm)
                }

                let inputData = Data(buffer: buffer)

                var error: Unmanaged<CFError>?
                let data = SecKeyCreateSignature(
                    self.key,
                    .ecdsaSignatureMessageX962SHA256,
                    inputData as CFData,
                    &error
                )
                if let error = error {
                    throw NarrativeSignError.signingFailed(error.takeRetainedValue())
                }

                guard let data = data else {
                    throw NarrativeSignError.invalidSignature
                }

                promise.succeed(ByteBuffer(data: data as Data))
            } catch {
                promise.fail(error)
            }
        }
        return promise.futureResult
    }

    // Decrypt is only used for RSA. All narrative keys should be used for EC.
    internal func decrypt(channel: Channel, data _: ByteBuffer) -> EventLoopFuture<ByteBuffer> {
        channel.eventLoop.makeFailedFuture(NarrativeSignError.unsupportedAlgorithm(.rsaPkcs1Sha256))
    }
}

internal enum NarrativeIdentityError: Error {
    case identityNotFound(String)
    case instantiationError(String)
    case certChainMissing(String)
    case privateKeyMissing(String)
}

internal func loadTLSCerts() async throws -> CloudMetricsCertConfig {
        logger.debug("Getting Narrative ACDC cert to establish mTLS")
    
        let acdcActorCert = NarrativeCert(domain: .acdc
                                      , identityType: .actor)
        guard let acdcActorCertRefs = acdcActorCert.fetchSecRefsFromKeychain() else {
            throw NarrativeHelperError.unableToGetCertificate("Error getting SecRefs for acdc actor certificate")
        }


        let certName  = getCertificateName(acdcActorCertRefs.certRef)
        logger.debug("""
                          ACDC actor cert details
                          Identifier=\(acdcActorCert.keychainLabel), \
                          CommonName=\(certName)
                    """)

        return try await getCloudMetricsCertConfigFromNarrativeIdentity(narrativeCert: acdcActorCert)
}

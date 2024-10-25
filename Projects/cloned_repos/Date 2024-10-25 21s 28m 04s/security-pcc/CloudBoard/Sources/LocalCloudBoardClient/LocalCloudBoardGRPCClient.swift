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

//  Copyright © 2024 Apple Inc. All rights reserved.

import CloudAttestation
@_implementationOnly import Crypto
import Foundation
@_implementationOnly import HTTPClientStateMachine
@_implementationOnly import InternalGRPC
import os

public enum LocalCloudBoardGRPCClientError: Error {
    case invalidPrivateCloudComputeResponse
    case invalidUUIDInCloudComputeResponse
    case noChunkInStreamResponse
    case unknownKeyType(String)
    case failedToParseCloudAttestationBundle(Error)
    case statusCode(Int)
}

public enum PrivateCloudComputeResponseMessage {
    case responseID(UUID)
    case payload(Data)
    case summary(String)
    case unknown
}

public class LocalCloudBoardGRPCClient {
    typealias CloudBoardGrpcClient = Com_Apple_Cloudboard_Api_V1_CloudBoardNIOClient
    typealias FetchAttestationRequest = Com_Apple_Cloudboard_Api_V1_FetchAttestationRequest
    typealias PrivateCloudComputeRequest = Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeRequest
    typealias PrivateCloudComputeResponse = Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse

    public static let logger: os.Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "LocalCloudBoardClient"
    )
    private let attestationEnvironment: CloudAttestation.Environment
    private var grpcClient: CloudBoardGrpcClient

    public init(host: String, port: Int) {
        self.attestationEnvironment = .dev
        self.grpcClient = Self.getGrpcClient(host: host, port: port)
    }

    public init(host: String, port: Int, attestationEnvironment: String) {
        self.attestationEnvironment = CloudAttestation.Environment(
            rawValue: attestationEnvironment
        ) ?? .dev
        self.grpcClient = Self.getGrpcClient(host: host, port: port)
    }

    deinit {
        _ = self.grpcClient.channel.close()
    }

    private static func getGrpcClient(
        host: String,
        port: Int
    ) -> CloudBoardGrpcClient {
        let group = PlatformSupport.makeEventLoopGroup(
            loopCount: 1,
            networkPreference: .userDefined(.networkFramework)
        )
        let queue = DispatchQueue(
            label: "com.apple.LocalCloudBoardClient.Verification.queue",
            qos: DispatchQoS.userInitiated
        )
        let channel = ClientConnection
            .usingTLSBackedByNetworkFramework(on: group)
            .withTLSHandshakeVerificationCallback(on: queue, verificationCallback: { _, _, verifyComplete in
                verifyComplete(true)
            })
            .connect(host: host, port: port)
        return CloudBoardGrpcClient(channel: channel)
    }

    public func submitPrivateRequest(payload: Data) async throws
    -> AsyncThrowingStream<PrivateCloudComputeResponseMessage, Error> {
        let pccAuthRequest = try PrivateCloudComputeRequest.serialized(with: .authToken(.init()))
        let pccPayloadRequest = try PrivateCloudComputeRequest.serialized(with: .applicationPayload(payload))

        // get the attestation
        let attestationResponse = try await grpcClient.fetchAttestation(
            FetchAttestationRequest()
        ).response.get()
        let cloudOSNodePublicKeyID = attestationResponse.attestation.keyID
        let (cloudOSNodePublicKey, bundleJson) = try await getPublicKeyFromAttestation(
            attestationResponse.attestation
                .attestationBundle
        )

        // create an ohttp request stream
        var oHTTPClientStateMachine = OHTTPClientStateMachine()
        let encapsulatedKey = try oHTTPClientStateMachine.encapsulateKey(
            keyID: 1,
            publicKey: cloudOSNodePublicKey,
            ciphersuite: .Curve25519_SHA256_AES_GCM_128
        )
        let encapsulatedTGT = try oHTTPClientStateMachine.encapsulateMessage(
            message: pccAuthRequest, isFinal: false
        )
        let encapuslatedPayload = try oHTTPClientStateMachine.encapsulateMessage(
            message: pccPayloadRequest, isFinal: true
        )

        // submit workload
        let stream = WorkloadClientStream(client: self.grpcClient)
        let (outputStream, outputStreamContinuation) = AsyncThrowingStream<PrivateCloudComputeResponseMessage, Error>
            .makeStream()
        stream.submitWorkload(
            keyID: cloudOSNodePublicKeyID,
            key: encapsulatedKey,
            bundleJson: bundleJson,
            chunks: [encapsulatedTGT, encapuslatedPayload]
        ) { result in
            switch result {
            case .success(let ResponseChunk):
                switch ResponseChunk {
                case .apiChunk(let chunk):
                    guard chunk.encryptedPayload.count > 0 else {
                        return
                    }

                    do {
                        var data = try oHTTPClientStateMachine.decapsulateResponseMessage(
                            chunk.encryptedPayload,
                            isFinal: chunk.isFinal
                        )
                        if let chunkData = data.readLengthPrefixedChunk() {
                            let pccResponse = try PrivateCloudComputeResponse(serializedData: chunkData)
                            let message: PrivateCloudComputeResponseMessage
                            switch pccResponse.type {
                            case .responseUuid(let uuidData):
                                if let uuid = UUID(from: uuidData) {
                                    message = PrivateCloudComputeResponseMessage.responseID(uuid)
                                } else {
                                    throw LocalCloudBoardGRPCClientError.invalidUUIDInCloudComputeResponse
                                }
                            case .responsePayload(let payloadData):
                                message = PrivateCloudComputeResponseMessage.payload(payloadData)
                            case .responseSummary(let responseSummary):
                                message = PrivateCloudComputeResponseMessage.summary(responseSummary.textFormatString())
                            default:
                                message = PrivateCloudComputeResponseMessage.unknown
                            }
                            outputStreamContinuation.yield(message)
                        }
                        if chunk.isFinal {
                            outputStreamContinuation.finish()
                        }
                    } catch {
                        outputStreamContinuation.finish(throwing: error)
                    }
                case .stringChunk(let StringChunk):
                    if let data = StringChunk.data(using: .utf8) {
                        let message = PrivateCloudComputeResponseMessage.payload(data)
                        outputStreamContinuation.yield(message)
                    } else {
                        outputStreamContinuation
                            .finish(throwing: LocalCloudBoardGRPCClientError.invalidPrivateCloudComputeResponse)
                    }
                }

            case .failure(let error):
                if let streamError = error as? WorkloadClientStreamError,
                   case .grpcNotOk(let status) = streamError {
                    let code = status.code.rawValue
                    outputStreamContinuation.finish(throwing: LocalCloudBoardGRPCClientError.statusCode(code))
                    return
                }
                outputStreamContinuation.finish(throwing: error)
            }
        }
        return outputStream
    }

    private func getPublicKeyFromAttestation(_ attestationBundle: Data) async throws
    -> (Curve25519.KeyAgreement.PublicKey, String?) {
        // https://www.ietf.org/archive/id/draft-thomson-http-oblivious-01.html#name-key-configuration-encoding
        let keyBytesHeaderSize = 3 // key ID (1) + KEM ID (2)
        let keyBytesSize = 32
        let keyBytesTrailerSize = 6 // length of trailer (2) + KDF ID (2) + AEAD ID (2)
        let expectedKeyConfigSize = keyBytesHeaderSize + keyBytesSize + keyBytesTrailerSize

        if attestationBundle.count == expectedKeyConfigSize {
            LocalCloudBoardGRPCClient.logger.log(
                "Received \(expectedKeyConfigSize, privacy: .public)-byte attestation bundle, interpreting as encoded key configuration"
            )
            let publicKey = try Curve25519.KeyAgreement.PublicKey(
                attestationBundle.subdata(
                    in: keyBytesHeaderSize ..< (attestationBundle.count - keyBytesTrailerSize)
                ),
                kem: .Curve25519_HKDF_SHA256
            )
            return (publicKey, nil)
        } else {
            LocalCloudBoardGRPCClient.logger.log("Received CloudAttestation attestation bundle. Extracting public key.")
            do {
                let bundle = try AttestationBundle(data: attestationBundle)
                let bundleJson = try bundle.jsonString()
                let validator = CloudAttestation.NodeValidator(environment: self.attestationEnvironment)
                let (key, validity, validatedAttestation) = try await validator.validate(bundle: bundle)
                LocalCloudBoardGRPCClient.logger.log(
                    "Verified attestation bundle with validity \(validity, privacy: .public) and expiration \(validatedAttestation.keyExpiration, privacy: .public), key: \(String(describing: key), privacy: .public)"
                )

                let rawKey: Data
                switch key {
                case .x963(let rawData):
                    rawKey = rawData
                case .curve25519(let rawData):
                    rawKey = rawData
                default:
                    LocalCloudBoardGRPCClient.logger.log("Unknown key type in attestation bundle")
                    throw LocalCloudBoardGRPCClientError.unknownKeyType(String(describing: key))
                }
                let publicKey = try Curve25519.KeyAgreement.PublicKey(rawKey, kem: .Curve25519_HKDF_SHA256)
                return (publicKey, bundleJson)
            } catch {
                LocalCloudBoardGRPCClient.logger.log(
                    "Failed to extract key from CloudAttestation attestation bundle: error (\(error, privacy: .public))"
                )
                throw LocalCloudBoardGRPCClientError.failedToParseCloudAttestationBundle(error)
            }
        }
    }
}

extension UUID {
    public init?(from data: Data) {
        guard data.count == MemoryLayout<uuid_t>.size else {
            return nil
        }

        let uuid: UUID? = data.withUnsafeBytes {
            guard let baseAddress = $0.bindMemory(to: UInt8.self).baseAddress else {
                return nil
            }
            return NSUUID(uuidBytes: baseAddress) as UUID
        }

        guard let uuid else {
            return nil
        }

        self = uuid
    }
}

extension HPKE.Ciphersuite {
    static var Curve25519_SHA256_AES_GCM_128: HPKE.Ciphersuite {
        .init(kem: .Curve25519_HKDF_SHA256, kdf: .HKDF_SHA256, aead: .AES_GCM_128)
    }
}

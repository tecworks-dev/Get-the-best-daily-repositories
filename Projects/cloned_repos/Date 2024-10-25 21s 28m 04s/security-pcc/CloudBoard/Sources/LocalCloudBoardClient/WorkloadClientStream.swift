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

// Copyright © 2024 Apple. All rights reserved.

import Combine
import Foundation
@_implementationOnly import InternalGRPC

typealias InvokeWorkloadRequest = Com_Apple_Cloudboard_Api_V1_InvokeWorkloadRequest
typealias InvokeWorkloadResponse = Com_Apple_Cloudboard_Api_V1_InvokeWorkloadResponse
typealias InvokeWorkloadClientStreamCall = BidirectionalStreamingCall<InvokeWorkloadRequest, InvokeWorkloadResponse>

enum CloudBoardResponseChunk {
    case apiChunk(Com_Apple_Cloudboard_Api_V1_Chunk)
    case stringChunk(String)
}

enum WorkloadClientStreamError: Error {
    case failedToCreateStream
    case grpcNotOk(GRPCStatus)
}

class WorkloadClientStream {
    private let client: LocalCloudBoardGRPCClient.CloudBoardGrpcClient

    public init(client: LocalCloudBoardGRPCClient.CloudBoardGrpcClient) {
        self.client = client
    }

    public func submitWorkload(
        keyID: Data,
        key: Data,
        bundleJson: String?,
        chunks: [Data],
        _ handler: @escaping (Result<CloudBoardResponseChunk, Error>) -> Void
    ) {
        var workloadStream: InvokeWorkloadClientStreamCall?
        workloadStream = self.client.invokeWorkload { response in
            handler(.success(.apiChunk(response.responseChunk)))
            if response.responseChunk.isFinal {
                workloadStream?.cancel(promise: nil)
            }
        }

        guard let workloadStream else {
            handler(.failure(WorkloadClientStreamError.failedToCreateStream))
            return
        }

        workloadStream.status.whenComplete {
            switch $0 {
            case .success(let status):
                if !status.isOk {
                    handler(.failure(WorkloadClientStreamError.grpcNotOk(status)))
                }
            default: break
            }
        }
        // if bundle json available, send back
        if let bundleJson {
            handler(.success(.stringChunk(bundleJson)))
        }

        do {
            // send setup request to start warming process
            try self.sendSetup(stream: workloadStream)

            // send parameters with decryption key
            try self.sendParameters(stream: workloadStream, keyID: keyID, key: key)

            // send request chunks
            for (index, chunk) in chunks.enumerated() {
                try self.sendChunk(stream: workloadStream, data: chunk, isFinal: index == chunks.count - 1)
            }
        } catch {
            handler(.failure(error))
        }
    }

    private func sendSetup(stream: InvokeWorkloadClientStreamCall) throws {
        let request = InvokeWorkloadRequest.with {
            $0.setup = Com_Apple_Cloudboard_Api_V1_Setup()
        }
        try stream.sendMessage(request).wait()
    }

    private func sendParameters(stream: InvokeWorkloadClientStreamCall, keyID: Data, key: Data) throws {
        let decryptionKey = Com_Apple_Cloudboard_Api_V1_DecryptionKey.with {
            $0.keyID = keyID
            $0.encryptedPayload = key
        }

        let parameters = Com_Apple_Cloudboard_Api_V1_Parameters.with {
            $0.decryptionKey = decryptionKey
            // set the tenant info values to clearly identify LocalCloudBoardClient traffic
            $0.tenantInfo = .with {
                $0.bundleID = "local-cloudboard-client"
                $0.automatedDeviceGroup = "local-test-device"
            }
        }

        let request = InvokeWorkloadRequest.with {
            $0.parameters = parameters
        }

        LocalCloudBoardGRPCClient.logger.log("Send parameters \((try? request.jsonString()) ?? "<unserialisable>")")
        try stream.sendMessage(request).wait()
    }

    private func sendChunk(stream: InvokeWorkloadClientStreamCall, data: Data, isFinal: Bool) throws {
        let requestChunk = Com_Apple_Cloudboard_Api_V1_Chunk.with {
            $0.encryptedPayload = data
            $0.isFinal = isFinal
        }
        let request = InvokeWorkloadRequest.with {
            $0.requestChunk = requestChunk
        }
        try stream.sendMessage(request).wait()
    }
}

extension Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeRequest {
    typealias TrustedCloudComputeRequest = Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeRequest
    typealias TrustedCloudComputeRequestType = TrustedCloudComputeRequest.OneOf_Type

    static func serialized(with type: TrustedCloudComputeRequestType) throws -> Data {
        var request = try TrustedCloudComputeRequest.with {
            $0.type = type
        }.serializedData()
        request.prependLength()
        return request
    }
}

extension Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse {
    typealias TrustedCloudComputeResponse = Com_Apple_Privatecloudcompute_Api_V1_PrivateCloudComputeResponse
    typealias TrustedCloudComputeResponseType = TrustedCloudComputeResponse.OneOf_Type

    static func serialized(with type: TrustedCloudComputeResponseType) throws -> Data {
        var request = try TrustedCloudComputeResponse.with {
            $0.type = type
        }.serializedData()
        request.prependLength()
        return request
    }
}

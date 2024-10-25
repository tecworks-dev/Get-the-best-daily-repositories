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

import CloudBoardJobAPI
import Foundation
import os

private let log: Logger = .init(
    subsystem: "com.apple.cloudos.cloudboard",
    category: "CloudBoardJob"
)

public struct InputData: AsyncSequence {
    let dataInputStream: AsyncStream<Data>
    public typealias Element = Data

    public struct AsyncIterator: AsyncIteratorProtocol {
        var iterator: AsyncStream<Data>.AsyncIterator

        public mutating func next() async throws -> Element? {
            await self.iterator.next()
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(iterator: self.dataInputStream.makeAsyncIterator())
    }
}

private typealias DataOutputContinuation = AsyncStream<Data>.Continuation
public struct ResponseWriter {
    fileprivate let outputContinuation: DataOutputContinuation
    fileprivate init(_ outputContinuation: DataOutputContinuation) {
        self.outputContinuation = outputContinuation
    }

    public func write(_ output: Data) async throws {
        self.outputContinuation.yield(output)
    }

    public func finish() throws {
        self.outputContinuation.finish()
    }
}

public protocol CloudBoardApp {
    init()
    func warmup() async throws
    func teardown() async throws
    func run(input: InputData, responder: ResponseWriter, environment: CloudAppEnvironment) async throws
}

extension CloudBoardApp {
    public func warmup() async throws {
        // Can be overridden by implementing type...
    }

    public func teardown() async throws {
        // Returning from this function signals acknowledgement of the
        // teardown request. The run() routine will be cancelled after this function
        // returns.
    }
}

extension CloudBoardApp {
    package static func bootstrap(
        server: CloudBoardJobAPIServerProtocol,
        metricsBuilder: CloudAppMetrics.Builder,
        preferencesOverride: CloudBoardJobConfig? = nil
    ) async throws {
        let preferences = if let preferencesOverride {
            preferencesOverride
        } else {
            try CloudBoardJobConfig.fromPreferences()
        }

        // Create the stream and ResponseWriter to handle output
        let (outputStream, outputStreamContinuation) = AsyncStream<Data>.makeStream()
        let responseWriter = ResponseWriter(outputStreamContinuation)

        // Create the stream and associated InputData to wrap input
        let (inputStream, inputStreamContinuation) = AsyncStream<Data>.makeStream()
        let inputData = InputData(dataInputStream: inputStream)

        // Create the stream and associated continuation to forward teardown requests
        let (teardownRequestStream, teardownRequestStreamContinuation) = AsyncStream<Void>.makeStream()

        let appInstance = Self()
        let waitForParametersEnabled = preferences.waitForParameters ?? true
        let xpcMessenger = await JobHelperMessenger(
            server: server,
            inputContinuation: inputStreamContinuation,
            teardownContinuation: teardownRequestStreamContinuation,
            log: log,
            appInstance: appInstance,
            metricsBuilder: metricsBuilder,
            waitForParametersEnabled: waitForParametersEnabled
        )

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await withThrowingTaskGroup(of: Void.self) { subGroup in
                    subGroup.addTask {
                        for await data in outputStream {
                            try await xpcMessenger.provideOutput(data)
                        }
                        try await xpcMessenger.endJob()
                    }

                    subGroup.addTask {
                        try await xpcMessenger.waitForWarmupComplete()

                        var parameters: ParametersData?
                        if waitForParametersEnabled {
                            parameters = try await xpcMessenger.waitForParameters()
                        }
                        let metrics = await xpcMessenger.buildMetrics()
                        do {
                            let environment = CloudAppEnvironment(
                                metrics: metrics,
                                plaintextMetadata: .init(parameters?.plaintextMetadata)
                            )
                            log.debug("""
                            CloudBoardJob invoked with: \
                            requestID=\(environment.plaintextMetadata.requestID, privacy: .public)
                            bundleID=\(environment.plaintextMetadata.bundleID, privacy: .public)
                            bundleVersion=\(environment.plaintextMetadata.bundleVersion, privacy: .public)
                            featureID=\(environment.plaintextMetadata.featureID, privacy: .public)
                            automatedDeviceGroup=\(environment.plaintextMetadata.automatedDeviceGroup, privacy: .public)
                            clientInfo=\(environment.plaintextMetadata.clientInfo, privacy: .public)
                            """)
                            try await appInstance.run(
                                input: inputData,
                                responder: responseWriter,
                                environment: environment
                            )
                            responseWriter.outputContinuation.finish()
                        } catch {
                            responseWriter.outputContinuation.finish()
                            log.error("""
                            Cloud app failed with error: \
                            \(String(reportable: error), privacy: .public) \
                            (\(error))
                            """)
                            throw error
                        }
                    }
                    try await subGroup.waitForAll()
                }
            }

            group.addTask {
                for await _ in teardownRequestStream {}
                try await appInstance.teardown()
            }

            // If one of these terminates, we want to cancel the other.
            _ = try await group.next()
            log.info("Cancelling all subtasks")
            await xpcMessenger.cancel()
            group.cancelAll()
        }
    }
}

extension CloudAppEnvironment.PlaintextMetadata {
    init(_ receivedData: ParametersData.PlaintextMetadata?) {
        if let receivedData {
            self.init(
                bundleID: receivedData.bundleID,
                bundleVersion: receivedData.bundleVersion,
                featureID: receivedData.featureID,
                clientInfo: receivedData.clientInfo,
                workloadType: receivedData.workloadType,
                workloadParameters: receivedData.workloadParameters,
                requestID: receivedData.requestID,
                automatedDeviceGroup: receivedData.automatedDeviceGroup
            )
        } else {
            self.init(
                bundleID: "",
                bundleVersion: "",
                featureID: "",
                clientInfo: "",
                workloadType: "",
                workloadParameters: [:],
                requestID: "",
                automatedDeviceGroup: ""
            )
        }
    }
}

enum CloudAppError: Error {
    case unsupported
}

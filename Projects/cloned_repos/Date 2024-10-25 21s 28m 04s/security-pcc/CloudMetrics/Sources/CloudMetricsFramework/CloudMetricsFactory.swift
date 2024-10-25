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
//  CloudMetricsFactory.swift
//  CloudMetricsFramework
//
//  Created by Andrea Guzzo on 8/25/22.
//

import Foundation
import os

internal typealias CloudMetricsClientCallback = () async -> Void
internal typealias CloudMetricsClientContinuation = AsyncStream<CloudMetricsClientCallback>.Continuation

private var logger = Logger(subsystem: kCloudMetricsLoggingSubsystem, category: "Factory")
internal let kCloudMetricsDispatchGroup = DispatchGroup()

internal final class CloudMetricsFactory: @unchecked Sendable {
    private var xpcClient: CloudMetricsXPCClient?
    private let clientStream: AsyncStream<CloudMetricsClientCallback>
    internal var clientStreamContinuation: AsyncStream<CloudMetricsClientCallback>.Continuation

    internal init() {
        // swiftlint:disable:next implicitly_unwrapped_optional
        var continuation: AsyncStream<CloudMetricsClientCallback>.Continuation! = nil
        self.clientStream = AsyncStream<CloudMetricsClientCallback> { continuation = $0 }
        self.clientStreamContinuation = continuation
        kCloudMetricsDispatchGroup.enter()
        Task {
            await connectXpcClient()
            for await callback in clientStream {
                if xpcClient == nil {
                    await connectXpcClient()
                }
                await callback()
            }
            kCloudMetricsDispatchGroup.leave()
        }
    }

    // This should be called only within the clientStream execution queue
    private func connectXpcClient() async {
        let continuation = self.clientStreamContinuation
        let client = CloudMetricsXPCClient(
            interruptionHandler: { _ in
                // interruptions are "soft errors", in theory we could retry sending messages
                // but given the best-effor nature of our service we can just ignore the event
                // and move forward.
            },
            invalidationHandler: { [weak self] _ in
                // Setting the client to nil will trigger renewal of the connection
                // at next metric update.
                // We want this to happen in the clientStream execution queue
                // to ensure thread safety
                continuation.yield {
                    logger.log("XPC connection invalidated")
                    self?.xpcClient = nil
                }
            })
        logger.log("Connecting to CloudMetrics service")
        await client.connect()
        self.xpcClient = client
    }

    internal func client() -> CloudMetricsXPCClient? {
        xpcClient
    }

    // MARK: - CloudMetrics Types which are not part of swift-metrics
    internal func makeHistogram(label: String, dimensions: [(String, String)], buckets: [Double]) -> HistogramHandler {
        let histogram = CloudMetricsHistogram(label: label,
                                              dimensions: dimensions.reduce(into: [:]) { $0[$1.0] = $1.1 },
                                              buckets: buckets)
        return CloudMetricsHistogramHandler(cloudMetrics: self, histogram: histogram)
    }

    internal func destroyHistogram(_ handler: HistogramHandler) {
    }

    internal func makeSummary(label: String, dimensions: [(String, String)], quantiles: [Double]) -> SummaryHandler {
        let summary = CloudMetricsSummary(label: label,
                                          dimensions: dimensions.reduce(into: [:]) { $0[$1.0] = $1.1 },
                                          quantiles: quantiles)
        return CloudMetricsSummaryHandler(cloudMetrics: self, summary: summary)
    }

    internal func destroySummary(_ handler: SummaryHandler) {
    }

    deinit {
        clientStreamContinuation.finish()
        _ = kCloudMetricsDispatchGroup.wait(timeout: .now() + .seconds(5))
    }
}

extension CloudMetricsFactory: MetricsFactory {
    /// Create a backing `CounterHandler`.
    ///
    /// - parameters:
    ///     - label: The label for the `CounterHandler`.
    ///     - dimensions: The dimensions for the `CounterHandler`.
    internal func makeCounter(label: String, dimensions: [(String, String)]) -> CounterHandler {
        let counter = CloudMetricsCounter(label: label,
                                          dimensions: dimensions.reduce(into: [:]) { $0[$1.0] = $1.1 })
        return CloudMetricsCounterHandler(cloudMetrics: self, counter: counter)
    }

    /// Create a backing `FloatingPointCounterHandler`.
    ///
    /// - parameters:
    ///     - label: The label for the `FloatingPointCounterHandler`.
    ///     - dimensions: The dimensions for the `FloatingPointCounterHandler`.
    internal func makeFloatingPointCounter(label: String, dimensions: [(String, String)])
        -> FloatingPointCounterHandler {
        let counter = CloudMetricsCounter(label: label,
                                          dimensions: dimensions.reduce(into: [:]) { $0[$1.0] = $1.1 })
        return CloudMetricsCounterHandler(cloudMetrics: self, counter: counter)
    }

    /// Create a backing `RecorderHandler`.
    ///
    /// - parameters:
    ///     - label: The label for the `RecorderHandler`.
    ///     - dimensions: The dimensions for the `RecorderHandler`.
    ///     - aggregate: Is data aggregation expected.
    internal func makeRecorder(label: String, dimensions: [(String, String)], aggregate: Bool) -> RecorderHandler {
        let recorder = CloudMetricsRecorder(label: label,
                                            dimensions: dimensions.reduce(into: [:]) { $0[$1.0] = $1.1 },
                                            aggregate: aggregate)
        return CloudMetricsRecorderHandler(cloudMetrics: self, recorder: recorder)
    }

    /// Create a backing `TimerHandler`.
    ///
    /// - parameters:
    ///     - label: The label for the `TimerHandler`.
    ///     - dimensions: The dimensions for the `TimerHandler`.
    internal func makeTimer(label: String, dimensions: [(String, String)]) -> TimerHandler {
        let timer = CloudMetricsTimer(label: label,
                                      dimensions: dimensions.reduce(into: [:]) { $0[$1.0] = $1.1 })
        return CloudMetricsTimerHandler(cloudMetrics: self, timer: timer)
    }

    /// Invoked when the corresponding `Counter`'s `destroy()` function is invoked.
    /// Upon receiving this signal the factory may eagerly release any resources related to this counter.
    ///
    /// - parameters:
    ///     - handler: The handler to be destroyed.
    internal func destroyCounter(_ handler: CounterHandler) {
    }

    /// Invoked when the corresponding `FloatingPointCounter`'s `destroy()` function is invoked.
    /// Upon receiving this signal the factory may eagerly release any resources related to this counter.
    ///
    /// - parameters:
    ///     - handler: The handler to be destroyed.
    internal func destroyFloatingPointCounter(_ handler: FloatingPointCounterHandler) {
    }

    /// Invoked when the corresponding `Recorder`'s `destroy()` function is invoked.
    /// Upon receiving this signal the factory may eagerly release any resources related to this recorder.
    ///
    /// - parameters:
    ///     - handler: The handler to be destroyed.
    internal func destroyRecorder(_ handler: RecorderHandler) {
    }

    /// Invoked when the corresponding `Timer`'s `destroy()` function is invoked.
    /// Upon receiving this signal the factory may eagerly release any resources related to this timer.
    ///
    /// - parameters:
    ///     - handler: The handler to be destroyed.
    internal func destroyTimer(_ handler: TimerHandler) {
    }
}

// Sendability is ensured by synchronising internally
extension CloudMetrics: @unchecked Sendable {}

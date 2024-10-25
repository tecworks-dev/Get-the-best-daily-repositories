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
//  CloudMetericsXPCClient.swift
//  
//
//  Created by Andrea Guzzo on 8/24/22.
//

import MantaAsyncXPC
import os

public enum CloudMetricsXPCClientError: Error {
    case internalError(String)
    case xpcError(String)
    case apiMisuse(String)
}

extension CloudMetricsXPCClientError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .internalError(let message):
            return "Internal Error: \(message)."
        case .xpcError(let message):
            return "XPC Error: \(message)."
        case .apiMisuse(let message):
            return "API misuse \(message)."
        }
    }
}

private let logger = Logger(subsystem: kCloudMetricsLoggingSubsystem, category: "XPCClient" )

// swiftlint:disable function_parameter_count
public actor CloudMetricsXPCClient: Sendable {
    public typealias ConnectionHandler = @Sendable (_: CloudMetricsXPCClient) -> Void
    private var connection: MantaAsyncXPCConnection?
    private var onInterruption: ConnectionHandler?
    private var onInvalidation: ConnectionHandler?
    private var configurationMessage: CloudMetricsServiceMessages.SetConfiguration?
    private var unallowedMetricLogTracker: [String: ContinuousClock.Instant] = [:]
    private let logThrottleIntervalSeconds: Int
    public init(interruptionHandler: @escaping ConnectionHandler, invalidationHandler: @escaping ConnectionHandler) {
        self.onInterruption = interruptionHandler
        self.onInvalidation = invalidationHandler
        if let defaults = UserDefaults(suiteName: kCloudMetricsPreferenceDomain) {
            let interval = defaults.integer(forKey: "AuditLogThrottleIntervalSeconds")
            logThrottleIntervalSeconds = interval > 0 ? interval : kCloudMetricsAuditLogThrottleIntervalDefault
        } else {
            logThrottleIntervalSeconds = kCloudMetricsAuditLogThrottleIntervalDefault
        }
    }

    public func connect() async {
        self.connection = await MantaAsyncXPCConnection.connect(to: kCloudMetricsXPCServiceName)
        await self.connection?.handleConnectionInvalidated { _ in
            await self.surpriseDisconnect()
            await self.onInvalidation?(self)
        }
        await connection?.handleConnectionInterrupted { connection in
            if let configurationMessage = await self.configurationMessage {
                do {
                    try await connection.send(configurationMessage)
                } catch {
                    logger.error("Unable to send configuration message on interruption: \(error, privacy: .public)")
                }
            }
            await self.onInterruption?(self)
        }
        await connection?.activate()
    }

    public func handleInterruption(callback: @escaping @Sendable (_: CloudMetricsXPCClient) -> Void) {
        onInterruption = callback
    }

    public func handleInvalidation(callback: @escaping @Sendable (_: CloudMetricsXPCClient) -> Void) {
        onInvalidation = callback
    }
}

extension CloudMetricsXPCClient {
    private func surpriseDisconnect() async {
        connection = nil
    }

    public func disconnect() async {
        if let connection = connection {
            await connection.handleConnectionInvalidated(handler: nil)
            await connection.cancel()
            self.connection = nil
        }
    }
}

extension CloudMetricsXPCClient: CloudMetricsClientProtocol {
    public func setConfiguration(_ configuration: CloudMetricsConfigurationDictionary) async throws {
        if self.configurationMessage != nil {
            // Another configuration messsage was already sent.
            throw CloudMetricsXPCClientError.apiMisuse("Configuration message already sent")
        }
        let configurationMessage = CloudMetricsServiceMessages.SetConfiguration(configuration)
        guard try await ((connection?.send(configurationMessage)) != nil) else {
            throw CloudMetricsXPCClientError.xpcError("Can't send SetConfifuration message")
        }
        self.configurationMessage = configurationMessage
    }

    private func logUnallowedMetric(_ metric: CloudMetric) {
        var shouldLog = true
        if let when = unallowedMetricLogTracker[metric.label] {
            let duration: Duration = .seconds(logThrottleIntervalSeconds)
            if when.duration(to: .now) < duration {
                shouldLog = false
            }
        }
        if shouldLog {
            logger.error("Metric \(metric.label) not allowed")
            unallowedMetricLogTracker[metric.label] = .now
        }
    }

    public func incrementCounter(_ counter: CloudMetricsCounter,
                                 by amount: Int64,
                                 epoch: Double) async throws {
        debugMetric(metric: counter, message: "incrementCounter", value: "\(amount)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.IncrementCounter(counter, by: amount, epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(counter.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(counter)
        }
    }

    public func incrementCounter(_ counter: CloudMetricsCounter,
                                 by amount: Double,
                                 epoch: Double) async throws {
        debugMetric(metric: counter, message: "incrementCounter", value: "\(amount)")
        guard let result = try await connection?.send(CloudMetricsServiceMessages.IncrementFloatingPointCounter(
            counter,
            by: amount,
            epoch: epoch
        )) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(counter.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(counter)
        }
    }

    public func resetCounter(_ counter: CloudMetricsCounter,
                             epoch: Double) async throws {
        debugMetric(metric: counter, message: "resetCounter", value: "")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.ResetCounter(counter, epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(counter.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(counter)
        }
    }

    public func recordInteger(recorder: CloudMetricsRecorder,
                              value: Int64,
                              epoch: Double) async throws {
        debugMetric(metric: recorder, message: "recordInteger", value: "\(value)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordInteger(recorder, value: value, epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(recorder.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(recorder)
        }
    }

    public func recordDouble(recorder: CloudMetricsRecorder,
                             value: Double,
                             epoch: Double) async throws {
        debugMetric(metric: recorder, message: "recordDouble", value: "\(value)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordDouble(recorder, value: value, epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(recorder.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(recorder)
        }
    }

    public func recordNanoseconds(timer: CloudMetricsTimer,
                                  duration: Int64,
                                  epoch: Double) async throws {
        debugMetric(metric: timer, message: "recordNanoseconds", value: "\(duration)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordNanoseconds(timer,
                                                                duration: duration,
                                                                epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(timer.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(timer)
        }
    }

    public func resetCounter(_ counter: CloudMetricsCounter, initialValue: Double, epoch: Double) async throws {
        debugMetric(metric: counter, message: "resetCounter", value: "")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.ResetCounterWithDoubleValue(counter,
                                                                          value: initialValue,
                                                                          epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(counter.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(counter)
        }
    }

    public func resetCounter(_ counter: CloudMetricsCounter, initialValue: Int64, epoch: Double) async throws {
        debugMetric(metric: counter, message: "resetCounter", value: "")
        guard try await ((connection?
            .send(CloudMetricsServiceMessages.ResetCounterWithIntValue(counter,
                                                                       value: initialValue,
                                                                       epoch: epoch)) != nil)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(counter.label)")
        }
    }

    public func recordInteger(histogram: CloudMetricsHistogram,
                              buckets: [Double],
                              value: Int64,
                              epoch: Double) async throws {
        debugMetric(metric: histogram, message: "recordInteger", value: "\(value)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordHistogramInteger(histogram,
                                                                     buckets: buckets,
                                                                     value: value,
                                                                     epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(histogram.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(histogram)
        }
    }

    public func recordDouble(histogram: CloudMetricsHistogram,
                             buckets: [Double],
                             value: Double,
                             epoch: Double) async throws {
        debugMetric(metric: histogram, message: "recordDouble", value: "\(value)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordHistogramDouble(histogram,
                                                                    buckets: buckets,
                                                                    value: value,
                                                                    epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(histogram.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(histogram)
        }
    }

    public func recordBuckets(histogram: CloudMetricsHistogram,
                              buckets: [Double],
                              bucketValues: [Int],
                              sum: Double,
                              count: Int,
                              epoch: Double) async throws {
        debugMetric(metric: histogram, message: "recordBuckets", value: "\(bucketValues)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordHistogramBuckets(histogram,
                                                                     buckets: buckets,
                                                                     values: bucketValues,
                                                                     sum: sum,
                                                                     count: count,
                                                                     epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(histogram.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(histogram)
        }
    }

    public func recordInteger(summary: CloudMetricsSummary,
                              quantiles: [Double],
                              value: Int64,
                              epoch: Double) async throws {
        debugMetric(metric: summary, message: "recordInteger", value: "\(value)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordSummaryInteger(summary,
                                                                   quantiles: quantiles,
                                                                   value: value,
                                                                   epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(summary.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(summary)
        }
    }

    public func recordDouble(summary: CloudMetricsSummary,
                             quantiles: [Double],
                             value: Double,
                             epoch: Double) async throws {
        debugMetric(metric: summary, message: "recordDouble", value: "\(value)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordSummaryDouble(summary,
                                                                  quantiles: quantiles,
                                                                  value: value,
                                                                  epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(summary.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(summary)
        }
    }

    public func recordQuantiles(summary: CloudMetricsSummary,
                                quantiles: [Double],
                                quantileValues: [Double],
                                sum: Double,
                                count: Int,
                                epoch: Double) async throws {
        debugMetric(metric: summary, message: "recordQauntiles", value: "\(quantileValues)")
        guard let result = try await connection?
            .send(CloudMetricsServiceMessages.RecordSummaryQuantiles(summary,
                                                                     quantiles: quantiles,
                                                                     values: quantileValues,
                                                                     sum: sum,
                                                                     count: count,
                                                                     epoch: epoch)) else {
            throw CloudMetricsXPCClientError.xpcError("Connection error while updating \(summary.label)")
        }
        if result == .notAllowed {
            logUnallowedMetric(summary)
        }
    }
}

private func debugMetric(metric: CloudMetric, message: String, value: String) {
    for debuggingPrefix in CloudMetrics.debugMetricPrefixArray() where metric.label.hasPrefix(debuggingPrefix) {
        // swiftlint:disable:next line_length
        logger.info("\(message, privacy: .public) [label: \(metric.label, privacy: .public), dimensions: \(metric.dimensions, privacy: .public), value: \(value, privacy: .public)]")
    }
}

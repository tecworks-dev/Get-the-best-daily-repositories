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

import CloudBoardMetrics
import Foundation

private let prefix = "cloudboard"

extension HistogramBuckets {
    fileprivate static let connectTime: Self = [
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
        20,
        40,
        80,
        160,
        320,
        640,
        1280,
        2560,
    ]

    fileprivate static let prewarmAllocateCount: HistogramBuckets = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]

    fileprivate static let requestTime: Self = [
        0.005524,
        0.007812,
        0.011049,
        0.015625,
        0.022097,
        0.03125,
        0.044194,
        0.0625,
        0.088388,
        0.125,
        0.176777,
        0.25,
        0.353553,
        0.5,
        0.707107,
        1.0,
        1.414214,
        1.681793,
        2.0,
        2.181015,
        2.378414,
        2.593679,
        2.828427,
        2.953652,
        3.084422,
        3.220981,
        3.363586,
        3.512504,
        3.668016,
        3.830413,
        4.0,
        4.177095,
        4.362031,
        4.555155,
        4.756828,
        4.967431,
        5.187358,
        5.417022,
        5.656854,
        5.907305,
        6.168843,
        6.441961,
        6.727171,
        7.025009,
        7.336032,
        7.660826,
        8.0,
        8.35419,
        8.724062,
        9.110309,
        9.513657,
        9.934862,
        10.374716,
        11.313708,
        12.337687,
        13.454343,
        14.672065,
        16.0,
        22.627417,
        32.0,
        45.254834,
        64.0,
        90.509668,
        128.0,
        181.019336,
        256.0,
        362.038672,
    ]
    fileprivate static let drainCompletionTime: Self = [
        0,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        320,
        340,
        360,
        380,
        400,
        800,
        1600,
    ]
    fileprivate static let waitForWarmupCompleteTime: Self = [
        0,
        0.0001,
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
        0.0010,
        0.0011,
        0.0012,
        0.0013,
        0.0014,
        0.0015,
        0.0016,
        0.0017,
        0.0018,
        0.0019,
        0.002,
        0.0025,
        0.003,
        0.0035,
        0.004,
        0.0045,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10,
        11,
        12,
        13,
        14,
        15,
        17.5,
        20,
        25,
        30,
    ]
    fileprivate static let requestFielderAllocationTime: Self = [
        0,
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10,
        11,
        12,
        13,
        14,
        15,
        17.5,
        20,
        25,
        30,
        60,
        90,
        120,
    ]
}

enum Metrics {
    enum CloudBoardDaemon {
        struct CBJobHelperExitCounter: ExitCounter {
            static let label: MetricLabel = "\(prefix)_cb_jobhelper_process_exit"
            var dimensions: MetricDimensions<DefaultExitDimensionKeys>
            var action: CounterAction
        }

        struct DrainCompletionTimeHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_drain_time_seconds"
            static let buckets: HistogramBuckets = .drainCompletionTime
            enum DimensionKey: String, RawRepresentable {
                case activeRequests
            }

            var dimensions: MetricDimensions<DimensionKey>
            var value: Double

            init(duration: Duration, activeRequests: Int) {
                self.value = duration.seconds
                self.dimensions = [
                    .activeRequests: "\(activeRequests)",
                ]
            }
        }
    }

    enum RequestFielder {
        struct CloudAppTerminateFailureCounter: Counter {
            static let label: MetricLabel = "\(prefix)_cloud_app_terminate_failure"
            var action: CounterAction
        }

        struct CBJobHelperTerminateFailureCounter: Counter {
            static let label: MetricLabel = "\(prefix)_cb_job_helper_terminate_failure"
            var action: CounterAction
        }
    }

    enum HotPropertiesController {
        struct HotPropertiesCanaryValue: Gauge {
            static var label: MetricLabel = "\(prefix)_hotproperties_canary"
            var value: Double
        }
    }

    enum ServiceDiscoveryPublisher {
        struct ReconnectCounter: Counter {
            static let label: MetricLabel = "\(prefix)_service_discovery_re_connect_total"
            var action: CounterAction
        }

        struct ConnectSucceededCounter: Counter {
            static let label: MetricLabel = "\(prefix)_service_discovery_connect_succeeded_total"
            var action: CounterAction
        }

        struct ConnectFailedCounter: ErrorCounter {
            static let label: MetricLabel = "\(prefix)_service_discovery_connect_failed_total"
            var dimensions: MetricDimensions<DefaultErrorDimensionKeys>
            var action: CounterAction
        }

        struct BackoffDurationHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_service_discovery_backoff_duration_in_seconds"
            static let buckets: HistogramBuckets = .connectTime
            var value: Double
        }

        struct SucceededConnectDuration: Histogram {
            static let label: MetricLabel = "\(prefix)_service_discovery_connect_succeeded_duration_in_seconds"
            static let buckets: HistogramBuckets = .connectTime
            var value: Double

            init(value: Duration) {
                self.value = value.seconds
            }
        }

        struct FailedConnectDuration: ErrorHistogram {
            static let label: MetricLabel = "\(prefix)_service_discovery_connect_failed_duration_in_seconds"
            static let buckets: HistogramBuckets = .connectTime

            var value: Double
            var dimensions: MetricDimensions<DefaultErrorDimensionKeys>
        }

        struct RegisteredServices: Gauge {
            static let label: MetricLabel = "\(prefix)_service_discovery_registered_services"

            var value: Int

            init(value: Int) {
                self.value = value
            }
        }
    }

    enum RequestFielderManager {
        struct PrewarmedInstancesDiedTotal: Counter {
            static let label: MetricLabel = "\(prefix)_prewarmed_instances_died_total"
            var action: CounterAction
        }

        struct PrewarmedPoolSizeGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_prewarm_request_fielder_pool_size"

            enum DimensionKey: String, RawRepresentable {
                case configuredSize
            }

            var dimensions: MetricDimensions<DimensionKey>
            var value: Int

            init(value: Int, configuredPoolSize: Int) {
                self.value = value
                self.dimensions = [
                    .configuredSize: "\(configuredPoolSize)",
                ]
            }
        }

        struct InvokeWarmupErrorTotal: Counter {
            static let label: MetricLabel = "\(prefix)_invoke_warmup_error_total"
            var action: CounterAction
        }

        struct FailedToAllocateTotal: Counter {
            static let label: MetricLabel = "\(prefix)_failed_to_allocate_request_fielder"
            enum DimensionKey: String, RawRepresentable {
                case retryCount
            }

            var dimensions: MetricDimensions<DimensionKey>
            var action: CounterAction

            init(action: CounterAction, retryCount: Int) {
                self.action = action
                self.dimensions = [
                    .retryCount: "\(retryCount)",
                ]
            }
        }

        struct TimeToAllocateRequestFielderHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_time_to_allocate_request_fielder_histogram"
            enum DimensionKey: String, RawRepresentable {
                case prewarmingEnabled
                case prewarmedPoolSize
            }

            static var buckets: HistogramBuckets = .requestFielderAllocationTime
            var dimensions: MetricDimensions<DimensionKey>
            var value: Double

            init(
                duration: Duration,
                prewarmingEnabled: Bool,
                prewarmedPoolSize: Int
            ) {
                self.value = duration.seconds
                self.dimensions = [
                    .prewarmingEnabled: "\(prewarmingEnabled)",
                    .prewarmedPoolSize: "\(prewarmedPoolSize)",
                ]
            }
        }

        struct WaitedForCreationTotal: Counter {
            static let label: MetricLabel = "\(prefix)_request_fielder_waited_for_creation_total"
            enum DimensionKey: String, RawRepresentable {
                case prewarmedPoolSize
            }

            var action: CounterAction
            var dimensions: MetricDimensions<DimensionKey>

            init(action: CounterAction, prewarmedPoolSize: Int) {
                self.action = action
                self.dimensions = [
                    .prewarmedPoolSize: "\(prewarmedPoolSize)",
                ]
            }
        }

        struct AttemptsToAllocateRequestFielderHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_attempts_to_allocate_request_fielder_histogram"
            static var buckets: HistogramBuckets = .prewarmAllocateCount
            var value: Int
        }

        struct FailedToCreateTotal: Counter {
            static let label: MetricLabel = "\(prefix)_request_fielder_create_failed_total"
            var action: CounterAction
        }

        struct ReclaimedPrewarmedTotal: Counter {
            static let label: MetricLabel = "\(prefix)_request_fielder_reclaimed_prewarmed_total"
            var action: CounterAction
        }

        struct FailedToReclaimTotal: Counter {
            static let label: MetricLabel = "\(prefix)_request_fielder_reclaim_failed_total"
            var action: CounterAction
        }

        struct PlaceholderDelegateWorkloadResponseInvoked: Counter {
            static let label: MetricLabel = "\(prefix)_placeholder_request_fielder_delegate_invoked_total"
            var action: CounterAction
        }
    }

    enum CloudBoardProvider {
        struct RequestCounter: Counter {
            static let label: MetricLabel = "\(prefix)_requests_total"
            enum DimensionKey: String, RawRepresentable {
                case errorDescription
                case automatedDeviceGroup
            }

            var action: CounterAction
            var dimensions: MetricDimensions<DimensionKey>

            init(action: CounterAction, automatedDeviceGroup: Bool) {
                self.action = action
                self.dimensions = [
                    .automatedDeviceGroup: automatedDeviceGroup.description,
                ]
            }
        }

        struct FailedRequestCounter: Counter {
            static let label: MetricLabel = "\(prefix)_requests_errors_total"
            enum DimensionKey: String, RawRepresentable {
                case errorDescription
                case automatedDeviceGroup
            }

            var action: CounterAction
            var dimensions: MetricDimensions<DimensionKey>

            init(action: CounterAction, failureReason: some Swift.Error, automatedDeviceGroup: Bool) {
                self.action = action
                self.dimensions = [
                    .errorDescription: String(reportable: failureReason),
                    .automatedDeviceGroup: automatedDeviceGroup.description,
                ]
            }
        }

        struct MaxConcurrentRequestCountExceededTotal: Counter {
            static let label: MetricLabel = "\(prefix)_max_concurrent_request_count_exceeded_total"
            var action: CounterAction
        }

        struct MaxConcurrentRequestCountRejectedTotal: Counter {
            static let label: MetricLabel = "\(prefix)_max_concurrent_request_count_rejected_total"
            var action: CounterAction
        }

        struct MaxCumulativeRequestSizeExceededTotal: Counter {
            static let label: MetricLabel = "\(prefix)_max_cumulative_request_size_exceeded_total"
            var action: CounterAction
        }

        struct RequestTimeHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_request_duration_seconds"
            static let buckets: HistogramBuckets = .requestTime
            var value: Double
            var dimensions: MetricDimensions<DimensionKey>

            enum Result: CustomStringConvertible {
                case success
                case error

                var description: String {
                    switch self {
                    case .success:
                        "success"
                    case .error:
                        "error"
                    }
                }
            }

            enum DimensionKey: String, RawRepresentable {
                case result
                case errorDescription
                case featureId
                case bundleId
                case automatedDeviceGroup
                case onlySetupReceived
            }

            init(
                duration: Duration,
                featureId: String?,
                bundleId: String?,
                automatedDeviceGroup: Bool,
                onlySetupReceived: Bool
            ) {
                self.value = duration.seconds
                self.dimensions = [
                    .result: "\(Result.success)",
                    .featureId: featureId ?? "",
                    .bundleId: bundleId ?? "",
                    .automatedDeviceGroup: automatedDeviceGroup.description,
                    .onlySetupReceived: onlySetupReceived.description,
                ]
            }

            init(
                duration: Duration,
                featureId: String?,
                bundleId: String?,
                automatedDeviceGroup: Bool,
                onlySetupReceived: Bool,
                failureReason: some Swift.Error
            ) {
                self.value = duration.seconds
                self.dimensions = [
                    .result: "\(Result.error)",
                    .featureId: featureId ?? "",
                    .bundleId: bundleId ?? "",
                    .automatedDeviceGroup: automatedDeviceGroup.description,
                    .onlySetupReceived: onlySetupReceived.description,
                    .errorDescription: String(reportable: failureReason),
                ]
            }
        }

        struct ConcurrentRequests: Gauge {
            static let label: MetricLabel = "\(prefix)_concurrent_requests"
            var value: Int
        }

        struct WaitForWarmupCompleteTimeHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_wait_for_warmup_complete_time_seconds"
            static let buckets: HistogramBuckets = .waitForWarmupCompleteTime
            var value: Double

            init(duration: Duration) {
                self.value = duration.seconds
            }
        }

        struct FailedWaitForWarmupCompleteTimeHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_failed_wait_for_warmup_complete_time_seconds"
            static let buckets: HistogramBuckets = .waitForWarmupCompleteTime
            var value: Double

            init(duration: Duration) {
                self.value = duration.seconds
            }
        }
    }

    enum SessionStore {
        struct SessionAddedCounter: Counter {
            static let label: MetricLabel = "\(prefix)_session_added_total"
            var action: CounterAction
        }

        struct SessionReplayedCounter: Counter {
            static let label: MetricLabel = "\(prefix)_session_replayed_total"
            var action: CounterAction
        }

        struct StoredSessions: Gauge {
            static let label: MetricLabel = "\(prefix)_stored_sessions"
            var value: Int
        }
    }

    enum HeartbeatPublisher {
        struct PublishedCounter: Counter {
            static let label: MetricLabel = "\(prefix)_heartbeat_publish_total"
            var action: CounterAction

            enum DimensionKey: String, RawRepresentable {
                case isHealthy
            }

            var dimensions: MetricDimensions<DimensionKey>

            init(action: CounterAction, isHealthy: Bool) {
                self.action = action
                self.dimensions = [
                    .isHealthy: "\(isHealthy)",
                ]
            }
        }

        struct FailedToPublishCounter: ErrorCounter {
            static let label: MetricLabel = "\(prefix)_heartbeat_publish_error_total"
            var dimensions: MetricDimensions<DefaultErrorDimensionKeys>
            var action: CounterAction
        }
    }

    enum TLSConfiguration {
        struct TLSVerificationErrorCounter: ErrorCounter {
            static let label: MetricLabel = "\(prefix)_tls_verification_failure_total"
            var dimensions: MetricDimensions<DefaultErrorDimensionKeys>
            var action: CounterAction

            init(action: CounterAction, failureReason: String) {
                self.init(dimensions: [.errorDescription: failureReason], action: action)
            }

            init(dimensions: MetricDimensions<DimensionKey>, action: CounterAction) {
                self.action = action
                self.dimensions = dimensions
            }
        }
    }

    enum AttestationProvider {
        struct AttestationTimeToExpiryGauge: Gauge {
            static var label: MetricLabel = "\(prefix)_attestation_time_to_expiry_seconds"
            var value: Double

            init(expireAt: Date) {
                self.value = expireAt.timeIntervalSince(.now)
            }
        }
    }
}

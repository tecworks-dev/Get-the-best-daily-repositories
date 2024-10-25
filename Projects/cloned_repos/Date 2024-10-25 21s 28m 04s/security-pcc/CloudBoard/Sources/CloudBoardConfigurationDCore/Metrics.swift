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

import CloudBoardMetrics

private let prefix = ConfigurationDaemon.metricsClientName

extension HistogramBuckets {
    fileprivate static let duration: HistogramBuckets = [
        1,
        5,
        15,
        30,
        60,
        600,
    ]
}

enum Metrics {
    enum Daemon {
        struct LaunchCounter: Counter {
            static let label: MetricLabel = "\(prefix)_daemon_launch_total"
            var action: CounterAction
        }

        struct LaunchNumberGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_daemon_launch_number"
            var value: Int
        }

        struct TimeSinceLastLaunchHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_daemon_time_since_last_launch"
            static let buckets: HistogramBuckets = .duration
            var value: Double
        }

        struct UptimeGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_daemon_uptime"
            var value: Int
        }

        struct ExitedLoopCounter: Counter {
            static let label: MetricLabel = "\(prefix)_daemon_exited_loop_total"
            var action: CounterAction
        }
    }

    enum Upstream {
        struct TotalRequestsCounter: Counter {
            static let label: MetricLabel = "\(prefix)_upstream_requests_total"
            var action: CounterAction
        }

        struct AnySuccessCounter: Counter {
            static let label: MetricLabel = "\(prefix)_upstream_requests_success_any"
            var action: CounterAction
        }

        struct SuccessUpToDateCounter: Counter {
            static let label: MetricLabel = "\(prefix)_upstream_requests_success_uptodate"
            var action: CounterAction
        }

        struct SuccessNewConfigCounter: Counter {
            static let label: MetricLabel = "\(prefix)_upstream_requests_success_newconfig"
            var action: CounterAction
        }

        struct FailureCounter: Counter {
            static let label: MetricLabel = "\(prefix)_upstream_requests_failure"
            var action: CounterAction
        }

        struct ConfigSizeGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_config_size_bytes"
            var value: Int
        }

        struct SuccessDurationHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_upstream_requests_success_duration_seconds"
            static let buckets: HistogramBuckets = .duration
            var value: Double
            init(durationSinceStart startInstant: ContinuousClock.Instant) {
                self.value = Double(secondsSinceStart: startInstant)
            }
        }

        struct IsMisconfiguredGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_upstream_is_misconfigured"
            var value: Int
        }

        struct IsUsingPlainHTTPGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_upstream_is_using_plain_http"
            var value: Int
        }
    }

    enum Cache {
        struct CacheStoreCounter: Counter {
            static let label: MetricLabel = "\(prefix)_cache_store"
            var action: CounterAction
        }

        struct CacheResetCounter: Counter {
            static let label: MetricLabel = "\(prefix)_cache_reset"
            var action: CounterAction
        }

        struct CacheHitCounter: Counter {
            static let label: MetricLabel = "\(prefix)_cache_hit"
            var action: CounterAction
        }

        struct CacheMissCounter: Counter {
            static let label: MetricLabel = "\(prefix)_cache_miss"
            var action: CounterAction
        }

        struct CacheSizeGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_cache_size"
            var value: Int
        }
    }

    enum Fetcher {
        struct TickCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_tick_count"
            var action: CounterAction
        }

        struct ErrorCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_error_count"
            var action: CounterAction
        }

        struct UpstreamRequestCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_upstream_request_count"
            var action: CounterAction
        }

        struct NewConfigFromUpstreamCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_new_config_from_upstream_count"
            var action: CounterAction
        }

        struct ConfigUpToDateCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_config_uptodate_count"
            var action: CounterAction
        }

        struct UpstreamFailureCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_upstream_failure_count"
            var action: CounterAction
        }

        struct NewConfigToRegistryCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_new_config_to_registry_count"
            var action: CounterAction
        }

        struct FallbackToRegistryCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_fallback_to_registry_count"
            var action: CounterAction
        }

        struct RegistryReplySuccessCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_new_config_registry_reply_success_count"
            var action: CounterAction
        }

        struct RegistryReplyFailureCounter: Counter {
            static let label: MetricLabel = "\(prefix)_fetcher_new_config_registry_reply_failure_count"
            var action: CounterAction
        }

        struct CurrentRevisionGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_fetcher_current_revision"
            enum DimensionKey: String {
                case revision
            }

            var dimensions: MetricDimensions<DimensionKey>
            var value: Double
            init(dimensions: MetricDimensions<DimensionKey>, value: Double) {
                self.dimensions = dimensions
                self.value = value
            }

            init(revision: String, isCurrent: Bool) {
                self.init(
                    dimensions: [
                        .revision: revision,
                    ],
                    value: isCurrent ? 1 : 0
                )
            }
        }

        struct IsRevisionGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_fetcher_is_revision"
            var value: Double
        }

        struct IsFallbackGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_fetcher_is_fallback"
            var value: Double
        }

        struct IsWaitingForFirstGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_fetcher_is_waiting_for_first"
            var value: Double
        }

        struct TickDurationHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_fetcher_tick_duration"
            static let buckets: HistogramBuckets = .duration
            var value: Double
            init(durationSinceStart startInstant: ContinuousClock.Instant) {
                self.value = Double(secondsSinceStart: startInstant)
            }
        }

        struct SleepDurationHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_fetcher_sleep_duration"
            static let buckets: HistogramBuckets = .duration
            var value: Double
        }

        struct RegistryApplyingSuccessDurationHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_fetcher_registry_applying_success_duration"
            static let buckets: HistogramBuckets = .duration
            var value: Double
            init(durationSinceStart startInstant: ContinuousClock.Instant) {
                self.value = Double(secondsSinceStart: startInstant)
            }
        }

        struct FirstDelayHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_fetcher_first_delay"
            static let buckets: HistogramBuckets = .duration
            var value: Double
            init(durationSinceStart startInstant: ContinuousClock.Instant) {
                self.value = Double(secondsSinceStart: startInstant)
            }
        }
    }

    enum Registry {
        struct ConnectionsGauge: Gauge {
            static let label: MetricLabel = "\(prefix)_registry_connections"
            var value: Double
        }

        struct NewConnectionCounter: Counter {
            static let label: MetricLabel = "\(prefix)_registry_new_connection"
            var action: CounterAction
        }

        struct DisconnectCounter: Counter {
            static let label: MetricLabel = "\(prefix)_registry_disconnected"
            var action: CounterAction
        }

        struct ApplyConfigToConnectionCounter: Counter {
            static let label: MetricLabel = "\(prefix)_registry_apply_config"
            var action: CounterAction
        }

        struct ApplyFallbackToConnectionCounter: Counter {
            static let label: MetricLabel = "\(prefix)_registry_apply_fallback"
            var action: CounterAction
        }

        struct ApplyingSuccessCounter: Counter {
            static let label: MetricLabel = "\(prefix)_registry_applying_success"
            var action: CounterAction
        }

        struct ApplyingFailureCounter: Counter {
            static let label: MetricLabel = "\(prefix)_registry_applying_failure"
            var action: CounterAction
        }

        struct ApplyingSuccessHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_registry_applying_success_duration"
            static let buckets: HistogramBuckets = .duration
            var value: Double
            init(durationSinceStart startInstant: ContinuousClock.Instant) {
                self.value = Double(secondsSinceStart: startInstant)
            }
        }

        struct ApplyingFailureHistogram: Histogram {
            static let label: MetricLabel = "\(prefix)_registry_applying_failure_duration"
            static let buckets: HistogramBuckets = .duration
            var value: Double
            init(durationSinceStart startInstant: ContinuousClock.Instant) {
                self.value = Double(secondsSinceStart: startInstant)
            }
        }
    }
}

extension Double {
    fileprivate init(secondsSinceStart startInstant: ContinuousClock.Instant) {
        self.init(startInstant.duration(to: .now).components.seconds)
    }
}

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
//  TC2ConfigurationIndex.swift
//  PrivateCloudCompute
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

package enum TC2ClientConfigurationKey: String, CaseIterable {
    case environment
    case customEnvironmentURL
    case customEnvironmentHost
    case ignoreCertificateErrors
    case rateLimitRequestTimeout
    case rateLimitRequestPath
    case rateLimitRequestMinimumSpacing
    case rateLimitUnmatchedRequestStorageTimeout
    case prefetchRequestTimeout
    case prefetchRequestPath
    case trustedRequestFirstPayloadChunkTimeout
    case trustedRequestPath
    case forceAEADKey
    case lttIssuer
    case ottIssuer
    case maxPrefetchedAttestations
    case maxTotalAttestations
    case maxInlineAttestations
    case prewarmAttestationsValidityInSeconds
    case maxPrefetchBatches
    case overrideCellID
    case rateLimiterSessionTimeout
    case rateLimiterSessionLengthForSoftening
    case rateLimiterDefaultJitterFactor
    case rateLimiterMaximumRateLimitTtl
    case rateLimiterMaximumRateLimitDuration
    case testSignalHeader
    case testOptions
    case useStructedTrustedRequest
    case allowedWorkloadParameters
    case proposedLiveOnEnvironment
    case bootFixedLiveOnEnvironment

    package var domain: String {
        switch self {
        case .environment:
            return "com.apple.privateCloudCompute"
        default:
            return "com.apple.privateCloudCompute.client"
        }
    }
}

package struct TC2ConfigurationIndex<Value> {
    package var domain: String
    package var name: String
    package var defaultValue: Value
    package var isAllowedOnCustomerBuilds: Bool

    init(key: TC2ClientConfigurationKey, _ defaultValue: Value, isAllowedOnCustomerBuilds: Bool = false) {
        self.domain = key.domain
        self.name = key.rawValue
        self.defaultValue = defaultValue
        self.isAllowedOnCustomerBuilds = isAllowedOnCustomerBuilds
    }
}

extension TC2ConfigurationIndex: Sendable where Value: Sendable {
}

extension TC2ConfigurationIndex {
    /// The environment to use. This defaults to "" to allow automatically picking carry vs. production.
    package static var environment: TC2ConfigurationIndex<String?> { .init(key: .environment, nil) }

    /// If set, this overrides the environment to use with a specific hostname.
    package static var customEnvironmentURL: TC2ConfigurationIndex<String?> { .init(key: .customEnvironmentURL, nil) }
    package static var customEnvironmentHost: TC2ConfigurationIndex<String?> { .init(key: .customEnvironmentHost, nil) }

    package static var ignoreCertificateErrors: TC2ConfigurationIndex<Bool> { .init(key: .ignoreCertificateErrors, false) }

    package static var rateLimitRequestTimeout: TC2ConfigurationIndex<Int> { .init(key: .rateLimitRequestTimeout, 30_000) }
    package static var rateLimitRequestPath: TC2ConfigurationIndex<String> { .init(key: .rateLimitRequestPath, "/ratelimits") }
    package static var rateLimitRequestMinimumSpacing: TC2ConfigurationIndex<Double> { .init(key: .rateLimitRequestMinimumSpacing, 60.0) }

    package static var prefetchRequestTimeout: TC2ConfigurationIndex<Int> { .init(key: .prefetchRequestTimeout, 30_000) }
    package static var prefetchRequestPath: TC2ConfigurationIndex<String> { .init(key: .prefetchRequestPath, "/prefetch") }

    package static var trustedRequestFirstPayloadChunkTimeout: TC2ConfigurationIndex<Int> { .init(key: .trustedRequestFirstPayloadChunkTimeout, 30_000) }
    package static var trustedRequestPath: TC2ConfigurationIndex<String> { .init(key: .trustedRequestPath, "/invoke") }

    package static var forceAEADKey: TC2ConfigurationIndex<String?> { .init(key: .forceAEADKey, nil) }

    package static var lttIssuer: TC2ConfigurationIndex<String> { .init(key: .lttIssuer, "tis.gateway.icloud.com") }
    package static var ottIssuer: TC2ConfigurationIndex<String> { .init(key: .ottIssuer, "rts.gateway.icloud.com") }

    package static var maxPrefetchedAttestations: TC2ConfigurationIndex<Int> { .init(key: .maxPrefetchedAttestations, 60) }
    package static var maxTotalAttestations: TC2ConfigurationIndex<Int> { .init(key: .maxTotalAttestations, 87) }
    package static var maxInlineAttestations: TC2ConfigurationIndex<Int> { .init(key: .maxInlineAttestations, 27) }
    package static var prewarmAttestationsValidityInSeconds: TC2ConfigurationIndex<Double> { .init(key: .prewarmAttestationsValidityInSeconds, 30.0 * 60.0) }
    package static var maxPrefetchBatches: TC2ConfigurationIndex<Int> { .init(key: .maxPrefetchBatches, 5) }

    package static var overrideCellID: TC2ConfigurationIndex<String?> { .init(key: .overrideCellID, nil, isAllowedOnCustomerBuilds: true) }

    package static var rateLimiterSessionTimeout: TC2ConfigurationIndex<Double> { .init(key: .rateLimiterSessionTimeout, 60.0) }
    package static var rateLimiterSessionLengthForSoftening: TC2ConfigurationIndex<Int> { .init(key: .rateLimiterSessionLengthForSoftening, 5) }
    package static var rateLimiterDefaultJitterFactor: TC2ConfigurationIndex<Double> { .init(key: .rateLimiterDefaultJitterFactor, 0.1) }
    package static var rateLimiterMaximumRateLimitTtl: TC2ConfigurationIndex<Double> { .init(key: .rateLimiterMaximumRateLimitTtl, 60.0 * 60.0 * 24.0) }
    package static var rateLimiterMaximumRateLimitDuration: TC2ConfigurationIndex<Double> { .init(key: .rateLimiterMaximumRateLimitDuration, 60.0 * 60.0 * 24.0) }
    package static var rateLimitUnmatchedRequestStorageTimeout: TC2ConfigurationIndex<Double> { .init(key: .rateLimitUnmatchedRequestStorageTimeout, 60.0) }

    /// If set, then /invoke requests to ropes will contain a header `apple-test-signal` with this value
    package static var testSignalHeader: TC2ConfigurationIndex<String?> { .init(key: .testSignalHeader, nil) }
    package static var testOptions: TC2ConfigurationIndex<String?> { .init(key: .testOptions, nil) }

    package static var allowedWorkloadParameters: TC2ConfigurationIndex<String> {
        .init(
            key: .allowedWorkloadParameters,
            [
                "model",
                "adapter",
                "input-token-count-interval-start-closed",
                "input-token-count-interval-end-open",
                "max-allowed-output-tokens-interval-start-closed",
                "max-allowed-output-tokens-interval-end-open",
            ].joined(
                separator: ","
            )
        )
    }

    package static var proposedLiveOnEnvironment: TC2ConfigurationIndex<String?> { .init(key: .proposedLiveOnEnvironment, nil) }
    package static var bootFixedLiveOnEnvironment: TC2ConfigurationIndex<String?> { .init(key: .bootFixedLiveOnEnvironment, nil) }
}

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
//  TC2Environment.swift
//  PrivateCloudCompute
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import FeatureFlags
import Foundation

package enum TC2EnvironmentFlags: FeatureFlagsKey {
    // On internal builds, this must be turned off to configure any
    // environment except the default of carry. No impact to customer
    // builds.
    case enforceEnvironment

    // Unused by privatecloudcomputed. The flag itself remains enabled in
    // our plist because it is still being read by CloudAttestation and
    // Transparency. We can presumably delete it at some point when pccd
    // is fully in control of environments.
    case productionEnvironmentAvailable
    package var domain: StaticString {
        return "PrivateCloudCompute"
    }
    package var feature: StaticString {
        switch self {
        case .enforceEnvironment: return "enforceEnvironment"
        case .productionEnvironmentAvailable: return "productionEnvironmentAvailable"
        }
    }
}

package enum TC2EnvironmentNames: String {
    case production = "production"
    case carry = "carry"
    case staging = "staging"
    case qa = "qa"
    case perf = "perf"
    case dev = "dev"
    case ephemeral = "ephemeral"
}

package enum TC2Environment {
    case production
    case carry
    case staging
    case qa
    case perf
    case dev
    case ephemeral

    package init?(name: String) {
        switch name {
        case TC2EnvironmentNames.production.rawValue: self = .production
        case TC2EnvironmentNames.carry.rawValue: self = .carry
        case TC2EnvironmentNames.staging.rawValue: self = .staging
        case TC2EnvironmentNames.qa.rawValue: self = .qa
        case TC2EnvironmentNames.perf.rawValue: self = .perf
        case TC2EnvironmentNames.dev.rawValue: self = .dev
        case TC2EnvironmentNames.ephemeral.rawValue: self = .ephemeral
        default: return nil
        }
    }

    package var name: String {
        switch self {
        case .production: return TC2EnvironmentNames.production.rawValue
        case .carry: return TC2EnvironmentNames.carry.rawValue
        case .staging: return TC2EnvironmentNames.staging.rawValue
        case .qa: return TC2EnvironmentNames.qa.rawValue
        case .perf: return TC2EnvironmentNames.perf.rawValue
        case .dev: return TC2EnvironmentNames.dev.rawValue
        case .ephemeral: return TC2EnvironmentNames.ephemeral.rawValue
        }
    }

    package var ropesHostname: String {
        switch self {
        case .production: return "ropes.apple.com"
        case .carry: return "ropes.apple.com"
        case .staging: return "ropes-staging.corp.apple.com"
        case .qa: return "ropes-qa.corp.apple.com"
        case .perf: return "ropes-perf.corp.apple.com"
        case .dev: return "ropes-dev.corp.apple.com"
        case .ephemeral: return "ropes-ephemeral.corp.apple.com"
        }
    }

    package var ropesUrl: URL {
        switch self {
        case .production: return URL(string: "https://ropes.apple.com/prod")!
        case .carry: return URL(string: "https://ropes.apple.com/carry")!
        case .staging: return URL(string: "https://ropes-staging.corp.apple.com")!
        case .qa: return URL(string: "https://ropes-qa.corp.apple.com")!
        case .perf: return URL(string: "https://ropes-perf.corp.apple.com")!
        case .dev: return URL(string: "https://ropes-dev.corp.apple.com")!
        case .ephemeral: return URL(string: "https://ropes-ephemeral.corp.apple.com")!
        }
    }

    package var configUrl: URL {
        switch self {
        case .production: return URL(string: "https://gateway-oblivious.apple.com/pcc/bag")!
        case .carry: return URL(string: "https://gateway-oblivious.apple.com/pcc/bag-carry")!
        case .staging: return URL(string: "https://gateway-oblivious-ic1.apple.com/pcc/bag-staging")!
        case .qa: return URL(string: "https://gateway-oblivious-ic1.apple.com/pcc/bag-qa")!
        case .perf: return URL(string: "https://gateway-oblivious-ic1.apple.com/pcc/bag-perf")!
        case .dev: return URL(string: "https://gateway-oblivious-ic1.apple.com/pcc/bag-dev")!
        case .ephemeral: return URL(string: "https://gateway-oblivious-ic1.apple.com/pcc/bag-ephemeral")!
        }
    }

    package var forceOHTTP: Bool {
        switch self {
        case .carry, .production: return true
        default: return false
        }
    }
}

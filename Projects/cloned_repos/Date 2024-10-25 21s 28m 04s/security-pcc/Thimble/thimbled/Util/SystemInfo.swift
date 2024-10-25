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
//  SystemInfo.swift
//  PrivateCloudCompute
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import CryptoKit
import Foundation
import MobileGestaltPrivate

package protocol SystemInfoProtocol {
    var bundleName: String? { get }
    var bundleVersion: String? { get }
    var bootSessionID: String { get }
    var uniqueDeviceID: String { get }
    var marketingProductName: String { get }
    var productName: String { get }
    var productVersion: String { get }
    var productType: String { get }
    var buildVersion: String { get }
}

package struct SystemInfo: SystemInfoProtocol {
    package var bundleName: String? {
        return Bundle.main.bundleIdentifier
    }

    package var bundleVersion: String? {
        return Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    }

    package var bootSessionID: String {
        return sysctl(name: "kern.bootsessionuuid")
    }

    package var uniqueDeviceID: String {
        #if os(macOS)
        // On macOS, the uniqueDeviceID exists but it does not
        // survive OS updates; however, this thing does.
        var uuid: uuid_t = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        var timeout: timespec = .init(tv_sec: 1, tv_nsec: 0)
        gethostuuid(&uuid, &timeout)
        return UUID(uuid: uuid).uuidString
        #else
        return MobileGestalt.current.uniqueDeviceID
        #endif
    }

    package var marketingProductName: String {
        // "macOS" / "iOS"
        return MobileGestalt.current.marketingProductName
    }

    package var marketingDeviceFamilyName: String {
        return MobileGestalt.current.marketingDeviceFamilyName
    }

    package var productName: String {
        // "macOS" / "iPhone OS"
        return MobileGestalt.current.productName
    }

    package var productVersion: String {
        // "14.6" / "18.0"
        return MobileGestalt.current.productVersion
    }

    package var productType: String {
        // "Mac15,8" / "iPhone14,5"
        return MobileGestalt.current.productType
    }

    package var buildVersion: String {
        // "23G42" / "22A282"
        return MobileGestalt.current.buildVersion
    }

    // MARK: - System Calls

    private func sysctl(name: String) -> String {
        var size = 0
        var rv = Darwin.sysctlbyname(name, nil, &size, nil, 0)
        guard rv == 0 else {
            let errstr = String(cString: strerror(errno))
            fatalError("sysctlbyname() returned \(rv): \(errstr)")
        }

        guard size > 0 else {
            fatalError("sysctlbyname() for key=\(name) returned size=\(size)")
        }

        var value: [CChar] = .init(repeating: 0, count: size)
        rv = Darwin.sysctlbyname(name, &value, &size, nil, 0)
        guard rv == 0 else {
            let errstr = String(cString: strerror(errno))
            fatalError("sysctlbyname() returned \(rv): \(errstr)")
        }

        return String(cString: value)
    }
}

extension SystemInfoProtocol {
    package var uniqueDeviceIDPercentile: Double {
        // There are at least two important properties of this
        // function. First, it gives a double that is uniformly
        // distributed in [0.0, 1.0). Second, it must be a pure
        // function of `uniqueDeviceID` that is stable; if it
        // changes between builds, the stability of the device
        // configuration will be compromised.

        // As a result, we only rely on things that are known
        // to be well defined and stable; such as SHA256.

        let uniqueDeviceID = self.uniqueDeviceID
        guard let uniqueDeviceIDData = uniqueDeviceID.data(using: .utf8) else {
            fatalError("uniqueDeviceIDPercentile failed to understand uniqueDeviceID utf8")
        }

        // The hash of the unique device ID gives a stable and random
        // enough 256 bits, which we truncate to build a random enough
        // Double with the desired properties. The trunaction actually
        // works out to be 52 bits.
        let digest = CryptoKit.SHA256.hash(data: uniqueDeviceIDData)
        let truncated = digest.withUnsafeBytes { buf in
            assert(buf.count == 32)
            return buf.load(as: UInt64.self)
        }

        // Fun with IEEE 754! We want to build a Double out of random
        // bits with a range of width 1, so we use the range [1, 2)
        // and subtract 1. If we set the sign to +, the exponent to
        // 0, then we are left with a fractional part that is random
        // in a number of the form "1.(fractional part)".

        // clear the sign bit, and the exponent high bit
        var bits = truncated & 0x3fff_ffff_ffff_ffff
        // set the other exponent bits, so exponent = 0 (1023 biased)
        bits = bits | 0x3ff0_0000_0000_0000
        // now we have: sign = 0, exponent = 0, fraction=(bits),
        // which gives a Double of 1.(bits), uniform in [1, 2).
        let result = Double(bitPattern: bits)
        assert(result.sign == .plus)
        assert(result.exponentBitPattern == 1023)
        // subtracting 1 gives a Double uniform in [0, 1).
        return result - 1
    }
}

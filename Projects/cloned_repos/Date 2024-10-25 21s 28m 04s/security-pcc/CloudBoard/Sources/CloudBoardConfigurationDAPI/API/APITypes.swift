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

import Foundation

/// Information about the current state of a peer's configuration.
public enum ConfigurationInfo: Codable, Sendable, Hashable, CustomStringConvertible {
    /// No configuration has been applied yet.
    case none

    /// The fallback configuration has been applied.
    case fallback

    /// The configuration with the provided revision has been applied.
    case revision(String)

    public var description: String {
        switch self {
        case .none:
            return "none"
        case .fallback:
            return "fallback"
        case .revision(let revision):
            return "revision: \(revision)"
        }
    }
}

/// A request to update the configuration of a peer.
public enum ConfigurationUpdate: Codable, Sendable, Hashable {
    /// No updated needed, the peer is in the desired state.
    case upToDate

    /// The peer needs to apply the fallback configuration.
    case applyFallback

    /// The peer needs to apply the provided configuration.
    case applyConfiguration(UnappliedConfiguration)
}

/// A request to register for configuration updates to a domain.
public struct Registration: Codable, Sendable, Hashable {
    /// The name of the domain to register for.
    public var domainName: String

    /// The current configuration of the caller.
    public var currentConfiguration: ConfigurationInfo

    /// Creates a new registration.
    /// - Parameters:
    ///   - domainName: The name of the domain to register for.
    ///   - currentConfiguration: The current configuration of the caller.
    public init(domainName: String, currentConfiguration: ConfigurationInfo) {
        self.domainName = domainName
        self.currentConfiguration = currentConfiguration
    }
}

/// The configuration package to apply on the application side.
///
/// To apply configuration means to accept the updated configuration and then report back that the new
/// configuration has been successfully accepted.
public struct UnappliedConfiguration: Codable, Sendable, Hashable {
    /// The identifier of the revision of the configuration package.
    public var revisionIdentifier: String

    /// The raw JSON contents of the configuration package for the requested domain.
    public var contentsJSON: Data

    /// Creates a new configuration package to apply.
    ///
    /// - Parameters:
    ///   - revisionIdentifier: The identifier of the revision of the configuration package.
    ///   - contentsJSON: The raw JSON contents of the configuration package for the requested domain.
    public init(revisionIdentifier: String, contentsJSON: Data) {
        self.revisionIdentifier = revisionIdentifier
        self.contentsJSON = contentsJSON
    }
}

/// A fallback to static configuration, when no dynamic configuration package is available.
public struct FallbackToStaticConfiguration: Codable, Sendable, Hashable {
    /// Creates a new fallback to static configuration.
    public init() {}
}

/// A successful result of applying a configuration package.
public struct ConfigurationApplyingSuccess: Codable, Sendable, Hashable {
    /// The identifier of the revision of the configuration package.
    public var revisionIdentifier: String

    /// Creates a new successful result of applying a configuration package.
    ///
    /// - Parameter revisionIdentifier: The identifier of the revision of the configuration package.
    public init(revisionIdentifier: String) {
        self.revisionIdentifier = revisionIdentifier
    }
}

/// A failure result of applying a configuration package.
public struct ConfigurationApplyingFailure: Codable, Sendable, Hashable {
    /// The identifier of the revision of the configuration package.
    public var revisionIdentifier: String

    /// Creates a new successful result of applying a configuration package.
    ///
    /// - Parameter revisionIdentifier: The identifier of the revision of the configuration package.
    public init(revisionIdentifier: String) {
        self.revisionIdentifier = revisionIdentifier
    }
}

/// Information about the current configuration version.
public struct ConfigurationVersionInfo: Codable, Sendable, Hashable {
    /// An enumeration that represents a version of a project.
    public enum Version: Codable, Sendable, Hashable {
        /// A concrete revision.
        case revision(String)

        /// Using fallback.
        case fallback
    }

    /// The last successfully applied version.
    public var appliedVersion: Version?

    /// Creates a new version info value.
    /// - Parameters:
    ///   - appliedVersion: The last successfully applied version.
    public init(
        appliedVersion: Version?
    ) {
        self.appliedVersion = appliedVersion
    }
}

extension ConfigurationVersionInfo.Version: CustomStringConvertible {
    public var description: String {
        switch self {
        case .revision(let string):
            return string
        case .fallback:
            return "fallback"
        }
    }
}

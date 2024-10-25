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
//  SplunkConfigWriter.swift
//  SecurityMonitorLite
//

import Foundation
@_weakLinked import SecureConfigDB

class SplunkConfigWriter {
    static let baseDirectory = URL(fileURLWithPath: "/var/db/securitymonitorlited/", isDirectory: true)
    static let configFileName = "splunk_config.plist"
    static let auditTableFileName = "audit_table.plist"

    static func configure() throws {
        try FileManager.default.createDirectory(at: self.baseDirectory, withIntermediateDirectories: true, attributes: nil)
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .xml

        let configURL = baseDirectory.appending(path: self.configFileName)
        let splunkConfigData = try encoder.encode(SplunkConfig())
        try splunkConfigData.write(to: configURL)

        let auditTableURL = baseDirectory.appending(path: self.auditTableFileName)
        let auditTable = AuditTable(
            formatStrings: [
                ProcessExecEvent.formatString,
                ProcessExitEvent.formatString,
                SSHLoginEvent.formatString,
                SSHLogoutEvent.formatString,
                IOKitOpenEvent.formatString,
                NWOpenEvent.formatString
            ]
        )
        let auditTableConfigData = try encoder.encode(auditTable)
        try auditTableConfigData.write(to: auditTableURL)
    }

    struct SplunkConfig: Encodable {
        static let ServerConfigParameterKey = "com.apple.securitymonitorlited.serverURL"
        static let ServerDefaultURL = "https://dzv2.apple.com"
        var server: String = SplunkConfig.ServerDefaultURL
        var indexName: String = "SecMonLite"
        // Splunkloggingd config wants a token, but the endpoint doesnt require one
        var token: String = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"
        var predicates: [String] = ["subsystem == \"\(SMLDaemon.daemonName)\" AND category == \"\(SMLDaemon.eventCategory)\""]
        var bufferSize: Int = 655536
        var bufferCount: Int = 1
        var level: String = "Default"
        var timeout: Int = 30
        var forwardCrashReports = false
        var forwardPanicReports = false

        enum CodingKeys: String, CodingKey {
            case server = "Server"
            case indexName = "Index"
            case predicates = "Predicates"
            case bufferSize = "BufferSize"
            case token = "Token"
            case bufferCount = "BufferCount"
            case level = "Level"
            case timeout = "Timeout"
            case forwardCrashReports = "ForwardCrashReports"
            case forwardPanicReports = "ForwardPanicReports"
        }

        init() {
            do {
                let configParams = try SecureConfigParameters.loadContents()

                // Check security mode, we don't allow changing this in customer mode
                guard configParams.securityPolicy != .customer else {
                    SMLDaemon.daemonLog.info("SecureConfigSecurityPolicy in customer mode, using \(SplunkConfig.ServerDefaultURL, privacy: .public)")
                    return
                }

                // Pull the configuration value down
                guard let configServerURL: String = try configParams.unvalidatedParameter(SplunkConfig.ServerConfigParameterKey) else {
                    SMLDaemon.daemonLog.info("Failed to fetch URL string from SecureConfigParameters, using \(SplunkConfig.ServerDefaultURL, privacy: .public)")
                    return
                }
                guard URL(string: configServerURL) != nil else {
                    SMLDaemon.daemonLog.error("SecureConfigParameters passed invalid URL: \(configServerURL, privacy: .public), using \(SplunkConfig.ServerDefaultURL, privacy: .public)")
                    return
                }

                SMLDaemon.daemonLog.log("Setting Splunk server URL to match SecureConfigParameters: \(configServerURL, privacy: .public)")
                self.server = configServerURL
            } catch {
                SMLDaemon.daemonLog.info("Failed to load SecureConfigParameters, using \(SplunkConfig.ServerDefaultURL, privacy: .public) Error: \(error, privacy: .public)")
            }
        }
    }

    struct AuditTable: Encodable {
        var senders: Subsystem
        enum CodingKeys: String, CodingKey {
            case senders = "Senders"
        }

        struct Subsystem: Encodable {
            var subsystemName: SenderData
            enum CodingKeys: String, CodingKey {
                case subsystemName = "securitymonitorlited"
            }
        }

        struct SenderData: Encodable {
            var formatStrings: [String: FormatStringData]
            enum CodingKeys: String, CodingKey {
                case formatStrings = "FormatStrings"
            }

            struct FormatStringData: Encodable {
                var state: String = "Allowed"
                var argNames: [String]
                enum CodingKeys: String, CodingKey {
                    case state = "AuditState"
                    case argNames = "ArgumentNames"
                }
            }
        }

        init(formatStrings: [String]) {
            self.senders = Subsystem(subsystemName: SenderData(formatStrings: [:]))
            for formatString in formatStrings {
                let structuredArgs = deriveArgsFromFormat(formatString: formatString)
                self.senders.subsystemName.formatStrings[formatString] = SenderData.FormatStringData(argNames: structuredArgs)
            }
        }

        func deriveArgsFromFormat(formatString: String) -> [String] {
            var tokens = formatString.components(separatedBy: " ")
            // First token is always event_name
            tokens.removeFirst()
            var structuredArgs = ["event_name"]
            for token in tokens where token.last == ":" {
                structuredArgs.append(String(token.dropLast()))
            }
            return structuredArgs
        }
    }
}

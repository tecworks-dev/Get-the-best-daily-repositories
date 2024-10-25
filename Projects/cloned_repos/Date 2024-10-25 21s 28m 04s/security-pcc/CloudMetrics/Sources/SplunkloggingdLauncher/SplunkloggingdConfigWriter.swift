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
//  SplunkloggingdConfigWriter.swift
//  SplunkloggingdConfigWriter
//
//  Created by Marco Magdy on 03/05/2024
//

import Foundation
import OSLog
import cloudOSInfo

internal let kDomain = "com.apple.prcos.splunkloggingd" as CFString

private let logger = Logger(subsystem: "SplunkloggingdConfigWriter", category: "")

@main
class SplunkloggingdConfigWriter {
    static func main() {
        // First command line argument is the path to the output configuration file
        if CommandLine.argc != 2 {
            logger.error("Usage: SplunkloggingdConfigWriter <output file>")
            exit(1)
        }
        let outputFile = CommandLine.arguments[1]
        let config = readConfig()
        if config == nil {
            logger.error("Failed to read configuration. No files will be written.")
            exit(1)
        }

        // Write the configuration to a plist file
        let url = URL(fileURLWithPath: outputFile)
        // Create the directory if it doesn't exist
        let directory = url.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
        } catch {
            logger.error("Failed to create directory \(directory.path). \(error)")
            exit(1)
        }

        do {
            let encoder = PropertyListEncoder()
            encoder.outputFormat = .xml
            let data = try encoder.encode(config)
            try data.write(to: url)
        } catch {
            logger.error("Failed to write configuration to file. \(error)")
            exit(1)
        }
        logger.info("Configuration written to \(url.path)")
    }
}

private func readConfig() -> SplunkloggingdConfiguration? {
    let result = SplunkloggingdConfiguration()
    // Read the server name
    let server = CFPreferencesCopyValue("Server" as CFString, kDomain, kCFPreferencesAnyUser, kCFPreferencesAnyHost)
    guard server != nil else {
        logger.error("Failed to read server from configuration. server is nil.")
        return nil
    }

    guard let server = server as? String else {
        logger.error("Failed to read server from configuration. server is not a string.")
        return nil
    }
    result.server = server

    // Read the index name
    let indexName = CFPreferencesCopyValue("Index" as CFString, kDomain, kCFPreferencesAnyUser, kCFPreferencesAnyHost)
    guard indexName != nil else {
        logger.error("Failed to read index from configuration. index is nil.")
        return nil
    }

    guard let indexName = indexName as? String else {
        logger.error("Failed to read index from configuration. index is not a string.")
        return nil
    }
    result.indexName = indexName

    // Read the token
    let token = CFPreferencesCopyValue("Token" as CFString, kDomain, kCFPreferencesAnyUser, kCFPreferencesAnyHost)
    if let token = token as? String {
        result.token = token
    }

    // Read the buffer size
    let bufferSize = CFPreferencesCopyValue("BufferSize" as CFString, kDomain, kCFPreferencesAnyUser, kCFPreferencesAnyHost)
    if bufferSize != nil {
        guard let bufferSize = bufferSize as? NSNumber else {
            logger.error("Failed to read buffer size from configuration. bufferSize is not a number.")
            return nil
        }
        result.bufferSize = Int64(truncating: bufferSize)
    }

    // Read the predicates array
    let predicates = CFPreferencesCopyValue("Predicates" as CFString, kDomain, kCFPreferencesAnyUser, kCFPreferencesAnyHost)
    guard predicates != nil else {
        logger.error("Failed to read predicates from configuration. predicates is nil.")
        return nil
    }
    guard let predicates = predicates as? [Any] else {
        logger.error("Failed to read predicates from configuration. predicates is not an array.")
        return nil
    }

    // Check that all elements in the array are strings
    for predicate in predicates {
        guard predicate is String else {
            logger.error("Failed to read predicates from configuration. One of the elements is not a string.")
            return nil
        }
    }

    result.predicates = predicates as! [String]

    // Read observability labels
    let cloudOSInfoProvider = CloudOSInfoProvider()
    do {
        result.observabilityLabels = try cloudOSInfoProvider.observabilityLabels()
    } catch {
        logger.warning("Failed to read observability labels. Proceeding without any labels.")
    }
    return result
}

internal class SplunkloggingdConfiguration: Encodable {
    var server: String = ""
    var indexName: String = ""
    var token: String?
    var predicates: [String] = []
    var bufferSize: Int64?
    var observabilityLabels: [String: String]?

    // serialize with PascalCase
    enum CodingKeys: String, CodingKey {
        case server = "Server"
        case indexName = "Index"
        case predicates = "Predicates"
        case bufferSize = "BufferSize"
        case token = "Token"
        case observabilityLabels = "GlobalLabels"
    }
}

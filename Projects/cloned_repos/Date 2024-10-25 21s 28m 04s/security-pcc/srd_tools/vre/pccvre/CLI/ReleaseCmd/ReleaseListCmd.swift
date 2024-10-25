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

//  Copyright © 2024 Apple, Inc. All rights reserved.
//

import ArgumentParserInternal
import Foundation

private let defaultReleaseCount: UInt = 10
private let maxReleaseCount: UInt = 100

extension CLI.ReleaseCmd {
    struct ReleaseListCmd: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "list",
            abstract: "Display list of releases in the Private Cloud Compute Transparency Log."
        )

        @OptionGroup var globalOptions: CLI.globalOptions
        @OptionGroup var swlogOptions: CLI.ReleaseCmd.options

        @Option(name: [.customLong("count"), .customShort("c")], help: "Releases to enumerate.")
        var reqCount: UInt = defaultReleaseCount

        @Option(name: [.customLong("startWindow"), .customLong("start")],
                help: "Start log search window (< 0 from last/end; >= 0 from first).")
        var startWindow: Int64?

        @Option(name: [.customLong("endWindow"), .customLong("end")],
                help: "End log search window")
        var endWindow: UInt64?

        @Flag(name: [.customLong("detail")],
              help: "Include details in output table (or complete payloads with --json).")
        var showDetails: Bool = false

        @Flag(name: [.customLong("json")],
              help: ArgumentHelp("Output json info.", visibility: .customerHidden))
        var jsonOutput: Bool = false

        mutating func validate() throws {
            guard reqCount <= maxReleaseCount else {
                throw ValidationError("maximum \(maxReleaseCount) releases.")
            }

            if reqCount == 0 {
                reqCount = defaultReleaseCount
            }
        }

        // releaseInfo provides simple representation of a SW Release entry from the Transparency Log
        //   for display as json output (and logging)
        private struct releaseInfo: Codable {
            struct tickets: Codable {
                var ap: String
                var code: [String] = []
                var data: [String] = []
            }

            let index: UInt64
            let dataHash: String
            let expireTime: UInt64 // unix epoch time
            let tickets: tickets
            let createTime: UInt64? // unix epoch time
            let rawData: Data?
            let metadata: String? // json

            init(
                _ rel: SWRelease,
                includePayloads: Bool = false // include tickets and metadata payloads in output
            ) {
                self.index = rel.index
                self.dataHash = rel.dataHash.hexString
                self.expireTime = rel.nodeData.expiryMs / 1000
                self.rawData = includePayloads ? rel.rawData : nil

                var relTickets = releaseInfo.tickets(
                    ap: rel.apManifest!.description
                )

                if let cryptexManifests = rel.cryptexManifests {
                    for cM in cryptexManifests {
                        if cM.isDataCryptex() {
                            relTickets.data.append(cM.description)
                        } else {
                            relTickets.code.append(cM.description)
                        }
                    }
                }

                self.tickets = relTickets
                var createTime: UInt64? = nil
                var metadata: String? = nil

                if let relMD = rel.metadata {
                    metadata = includePayloads ? try? relMD.jsonString() : nil
                    if let pDate = relMD.timestamp {
                        createTime = UInt64(pDate.timeIntervalSince1970)
                    }
                }

                self.createTime = createTime
                self.metadata = metadata
            }
        }

        func run() async throws {
            CLI.setupDebugStderr(debugEnable: globalOptions.debugEnable)

            let logEnvironment = swlogOptions.environment
            CLI.logger.log("release list")
            if logEnvironment != .production {
                CLI.logger.log("using KT environment: \(logEnvironment.rawValue)")
            }

            var swlog = try await SWReleases(
                environment: logEnvironment,
                altKtInitEndpoint: swlogOptions.ktInitEndpoint,
                tlsInsecure: swlogOptions.tlsInsecure
            )

            try await swlog.fetchReleases(
                reqCount: reqCount,
                startWindow: startWindow,
                endWindow: endWindow
            )

            guard swlog.count > 0 else {
                throw CLIError("no matching releases found")
            }

            let relCount = min(UInt(swlog.count), reqCount)
            if !jsonOutput {
                print("Found \(relCount) release\(relCount != 1 ? "s" : ""):\n")
            }

            var swReleases: [releaseInfo] = []
            for rel in swlog[..<Int(relCount)] {
                let relInfo = releaseInfo(rel, includePayloads: showDetails)
                swReleases.append(relInfo)

                if jsonOutput { // for --json output, print everything at the end
                    continue
                }

                print("\(String(format: "%8d", rel.index)): \(relInfo.dataHash)")

                if showDetails {
                    print("          [expires: \(dateAsString(relInfo.expireTime))]")
                    if let pDate = relInfo.createTime {
                        print("          [created: \(dateAsString(pDate))]")
                    } else {
                        // no metadata available
                        print("          [not published]")
                    }

                    print("     Tickets")
                    print("           OS: \(relInfo.tickets.ap)")
                    if !relInfo.tickets.code.isEmpty || !relInfo.tickets.data.isEmpty {
                        print("       Cryptexes")
                        for c in relInfo.tickets.code {
                            print("         Code: \(c)")
                        }
                        for c in relInfo.tickets.data {
                            print("         Data: \(c)")
                        }
                    }

                    print()
                }
            }

            CLI.logger.log("total matching releases: \(relCount, privacy: .public)")
            CLI.logger.debug("displayed releases: \(asJSONString(swReleases), privacy: .public)")

            if jsonOutput {
                print(asJSONString(swReleases))
            } else if !showDetails {
                print()
            }
        }
    }
}

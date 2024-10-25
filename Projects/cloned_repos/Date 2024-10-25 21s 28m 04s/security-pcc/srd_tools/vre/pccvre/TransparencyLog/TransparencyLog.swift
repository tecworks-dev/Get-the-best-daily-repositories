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
import CloudAttestation
import Foundation
import os

// allow ArgumentParser to map it to command-line arg
extension CloudAttestation.Environment: @retroactive ExpressibleByArgument {}

// TransparencyLog provides methods to retrieve log entries from KT Auditor log
struct TransparencyLog: Sendable {
    typealias Environment = CloudAttestation.Environment

    
    var ktInitURL: URL? {
        switch self.environment {
            case .production: return URL.normalized(string: "init-kt-prod.ess.apple.com")
            case .carry: return URL.normalized(string: "init-kt-carry.ess.apple.com")
            case .qa, .staging: return URL.normalized(string: "init-kt-qa1.ess.apple.com")
            case .qa2Primary, .qa2Internal: return URL.normalized(string: "init-kt-qa2.ess.apple.com")
            default: return nil
        }
    }

    var application: TxPB_Application {
        self.environment.transparencyPrimaryTree ? .privateCloudCompute : .privateCloudComputeInternal
    }

    static let atLogType: TxPB_LogType = .atLog
    static let requestUUIDHeader = "X-Apple-Request-UUID"

    static let logger = os.Logger(subsystem: applicationName, category: "TransparencyLog")
    static var traceLog: Bool = false

    let environment: Environment
    let tlsInsecure: Bool // don't verify certificates; use "http:" in URI to skip TLS altogether
    let traceLog: Bool // include (excessive) debugging messages of Transparency Log calls
    let instanceUUID: UUID // passed into upstream API (for logging)
    var ktInitBag: KTInitBag? // KT Init Bag (links to Transparency endpoints for selected env/application)

    init(
        environment: Environment,
        altKtInitEndpoint: URL? = nil,
        tlsInsecure: Bool = false,
        loadKtInitBag: Bool = true,
        traceLog: Bool = false
    ) async throws {
        self.environment = environment
        self.tlsInsecure = tlsInsecure
        self.traceLog = traceLog
        self.instanceUUID = UUID()

        let uuidString = self.instanceUUID.uuidString
        TransparencyLog.logger.debug("Session UUID: \(uuidString, privacy: .public)")

        // load KTInitBag for the other endpoints
        if loadKtInitBag {
            guard let endpoint = altKtInitEndpoint ?? self.ktInitURL else {
                throw TransparencyLogError("must provide KT Init Bag endpoint")
            }

            TransparencyLog.logger.debug("Using KT Init endpoint: \(endpoint.absoluteString, privacy: .public)")

            do {
                self.ktInitBag = try await KTInitBag(
                    endpoint: endpoint,
                    tlsInsecure: self.tlsInsecure
                )
            } catch {
                throw TransparencyLogError("Fetch KT Init Bag: \(error)")
            }

            if traceLog {
                self.ktInitBag?.debugDump()
            }
        }
    }
}

// functions to collect layers to get us to enumerating log leaves for this instance
extension TransparencyLog {
    // fetchPubkeys retrieves public keys used to verify signature of Top Level trees
    @discardableResult
    func fetchPubkeys(
        altEndpoint: URL? = nil,
        tlsInsecure: Bool? = nil
    ) async throws -> PublicKeys {
        guard let endpoint = altEndpoint ?? ktInitBag?.url(.atResearcherPublicKeys) else {
            throw TransparencyLogError("must provide Public Keys endpoint")
        }

        TransparencyLog.logger.debug("Using KT Public Keys endpoint: \(endpoint.absoluteString, privacy: .public)")
        do {
            let pubKeys = try await PublicKeys(
                endpoint: endpoint,
                tlsInsecure: tlsInsecure ?? self.tlsInsecure,
                requestUUID: self.instanceUUID
            )

            return pubKeys
        } catch {
            throw TransparencyLogError("Fetch Public Keys: \(error)")
        }
    }

    // fetchLogTree retrieves top-level trees and selects active one for Private Cloude Compute application
    @discardableResult
    func fetchLogTree(
        logType: TxPB_LogType = Self.atLogType,
        application: TxPB_Application? = nil,
        altEndpoint: URL? = nil,
        tlsInsecure: Bool? = nil
    ) async throws -> Tree {
        guard let endpoint = altEndpoint ?? ktInitBag?.url(.atResearcherListTrees) else {
            throw TransparencyLogError("must provide List Keys endpoint")
        }

        TransparencyLog.logger.debug("Using KT List Trees endpoint: \(endpoint.absoluteString, privacy: .public)")

        var logTrees: TransparencyLog.Trees
        do {
            logTrees = try await Trees(
                endpoint: endpoint,
                tlsInsecure: tlsInsecure ?? self.tlsInsecure,
                requestUUID: self.instanceUUID
            )
        } catch {
            throw TransparencyLogError("Fetch Log Trees: \(error)")
        }

        if self.traceLog {
            logTrees.debugDump()
        }

        guard let relTree = logTrees.select(
            logType: logType,
            application: application ?? self.application
        ) else {
            throw TransparencyLogError("PCC Log Tree not found [\(application ?? self.application)]")
        }

        TransparencyLog.logger.debug("Using Private Cloud Tree ID [\(relTree.treeID, privacy: .public)]")
        return relTree
    }

    // fetchLogHead retrieves the log head of the application ("releases") tree
    @discardableResult
    func fetchLogHead(
        logTree: Tree,
        useCache: Bool = false,
        altEndpoint: URL? = nil,
        tlsInsecure: Bool? = nil
    ) async throws -> Head {
        let logger = TransparencyLog.logger
        guard let endpoint = altEndpoint ?? ktInitBag?.url(.atResearcherLogHead) else {
            throw TransparencyLogError("must provide Log Head endpoint")
        }

        TransparencyLog.logger.debug("Using Log Head endpoint: \(endpoint.absoluteString, privacy: .public)")

        do {
            let relLogHead = try await Head(
                endpoint: endpoint,
                tlsInsecure: tlsInsecure ?? self.tlsInsecure,
                logTree: logTree,
                appCerts: nil, 
                requestUUID: self.instanceUUID
            )
            if self.traceLog {
                logger.debug("LogHead: log size: \(relLogHead.size, privacy: .public); revision: \(relLogHead.revision, privacy: .public)")
            }
            return relLogHead
        } catch {
            throw TransparencyLogError("Fetch Log Head for PCC: \(error)")
        }
    }

    func fetchATLogLeaves(
        logTree: Tree,
        logHead: Head,
        reqCount: UInt = 10, // requested number of matching leaves
        startWindow: Int64? = nil, // log index search windows
        endWindow: UInt64? = nil,
        windowSize: UInt64 = 100,
        nodeDataType: ATLeafType? = nil, // eg .release node
        altEndpoint: URL? = nil,
        tlsInsecure: Bool? = nil
    ) async throws -> ([ATLeaf], startIndex: UInt64, endIndex: UInt64) {
        guard let endpoint = altEndpoint ?? ktInitBag?.url(.atResearcherLogLeaves) else {
            throw TransparencyLogError("must provide Log Leaves endpoint")
        }

        TransparencyLog.logger.debug("Using Log Leaves endpoint: \(endpoint.absoluteString, privacy: .public)")

        let maxIndex = logHead.size
        var endIndex = UInt64(min(maxIndex, endWindow ?? maxIndex))
        var startIndex: UInt64

        if let startWindow {
            startIndex = startWindow < 0 ? ((endIndex > -startWindow) ? endIndex - UInt64(-startWindow) : 0) :
                UInt64(startWindow)
        } else {
            startIndex = (endIndex > windowSize) ? endIndex - windowSize + 1 : 0
        }

        guard startIndex < endIndex else {
            throw TransparencyLogError("invalid index range")
        }

        let logLeaves = TransparencyLog.Leaves(
            endpoint: endpoint,
            tlsInsecure: tlsInsecure ?? self.tlsInsecure,
            logTree: logTree
        )

        var atLeaves: [ATLeaf] = []
        repeat {
            do {
                let leaves = try await logLeaves.fetch(startIndex: startIndex,
                                                       endIndex: endIndex,
                                                       requestUUID: self.instanceUUID,
                                                       nodeDecoder: {
                                                           guard let atleaf = ATLeaf($0) else {
                                                               return nil as ATLeaf?
                                                           }

                                                           if let nodeDataType {
                                                               guard atleaf.nodeData.type == nodeDataType else {
                                                                   return nil as ATLeaf?
                                                               }
                                                           }

                                                           return atleaf
                                                       })
                atLeaves.append(contentsOf: leaves)
            } catch {
                throw TransparencyLogError("Fetch Log Entries [\(startIndex)..\(endIndex)] for PCC: \(error)")
            }

            endIndex = endIndex > windowSize ? endIndex - windowSize : 0
            startIndex = startIndex > windowSize ? startIndex - windowSize : 0
        } while startIndex < endIndex && atLeaves.count < reqCount

        return (atLeaves, startIndex, endIndex)
    }

    // MARK: - Generic leaf retrieval

    func fetchLogLeaves<L: Leaf>(type: L.Type,
                                 tree: Tree,
                                 start: UInt64? = nil,
                                 end: UInt64? = nil) async throws -> [L]
    {
        try await self.fetchLogLeaves(type: type,
                                      tree: tree,
                                      head: self.fetchLogHead(logTree: tree),
                                      start: start,
                                      end: end)
    }

    func fetchLogLeaves<L: Leaf>(type: L.Type,
                                 tree: Tree,
                                 head: Head,
                                 start: UInt64? = nil,
                                 end: UInt64? = nil,
                                 batchSize: UInt64 = 3000) async throws -> [L]
    {
        guard let endpoint = ktInitBag?.url(.atResearcherLogLeaves) else {
            throw TransparencyLogError("must provide Log Leaves endpoint")
        }

        let logger = TransparencyLog.logger
        logger.debug("Using Log Leaves endpoint: \(endpoint.absoluteString)")

        let maxIndex = head.size
        let endIndex = UInt64(min(maxIndex, end ?? maxIndex))
        let startIndex: UInt64 = start ?? 0

        guard startIndex < endIndex else {
            return []
        }

        let logLeaves = TransparencyLog.Leaves(
            endpoint: endpoint,
            tlsInsecure: self.tlsInsecure,
            logTree: tree
        )

        var outLeaves: [L] = []
        var currentStart = startIndex
        var currentEnd = min(startIndex + batchSize, endIndex)
        repeat {
            do {
                let leaves = try await logLeaves.fetch(startIndex: currentStart, endIndex: currentEnd, requestUUID: self.instanceUUID, nodeDecoder: {
                    guard let leaf = L($0) else {
                        return nil as L?
                    }
                    return leaf as L?
                })
                outLeaves.append(contentsOf: leaves)
                currentStart += batchSize
                currentEnd = min(currentEnd + batchSize, endIndex)
            } catch {
                throw TransparencyLogError("Fetch Log Entries [\(startIndex)..<\(endIndex)] for PCC: \(error)")
            }
        } while currentStart < currentEnd
        return outLeaves
    }
}

// TransparencyLogError provides general error encapsulation for errors encountered when interacting
//  with the TransparencyLog
struct TransparencyLogError: Error, CustomStringConvertible {
    var message: String
    var description: String { self.message }

    init(_ message: String) {
        TransparencyLog.logger.error("\(message, privacy: .public)")
        self.message = message
    }
}

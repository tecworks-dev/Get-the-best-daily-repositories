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
//  TC2AttestationStore.swift
//  PrivateCloudCompute
//
//  Copyright © 2024 Apple Inc. All rights reserved.
//

import Foundation
import PrivateCloudCompute
import SwiftData
import os.lock

protocol TC2AttestationStoreProtocol: Sendable {
    func saveValidatedAttestation(
        _ validatedAttestation: ValidatedAttestation,
        for parameters: TC2RequestParameters,
        prefetched: Bool,
        batch: UInt,
        fetchTime: Date
    ) async -> Bool
    func getAttestationsForRequest(
        parameters: TC2RequestParameters,
        serverRequestID: UUID,
        maxAttestations: Int
    ) async -> [String: ValidatedAttestation]
    func deleteAllAttestationStoreEntries() async
    func deleteEntriesWithExpiredAttestationBundles() async
    func deleteEntryForNode(nodeIdentifier: String) async -> Bool
    func deleteEntries(
        withParameters parameters: TC2RequestParameters,
        batchId: UInt
    ) async
    func nodeExists(
        withUniqueIdentifier uniqueIdentifier: String
    ) async -> Bool
    func trackNodeForParameters(
        forParameters parameters: TC2RequestParameters,
        withUniqueIdentifier uniqueIdentifier: String,
        prefetched: Bool,
        batchID: UInt,
        fetchTime: Date
    ) async -> Bool
    func attestationsExist(
        forParameters: TC2RequestParameters,
        clientCacheSize: Int,
        fetchTime: Date
    ) async -> Bool
    func deleteAttestationsUsedByTrustedRequest(
        serverRequestID: UUID
    ) async -> UInt
    func getAttestationBundlesUsedByTrustedRequest(
        serverRequestID: UUID
    ) async -> [String: Data]
}

private let logger = tc2Logger(forCategory: .AttestationStore)

typealias TC2ParamsStoreEntry = TC2AttestationStoreMigrationPlan.TC2AttestationStoreSchema_v1.TC2ParamsStoreEntry
typealias TC2NodeStoreEntry = TC2AttestationStoreMigrationPlan.TC2AttestationStoreSchema_v1.TCNodeStoreEntry

public enum TC2AttestationStoreMigrationPlan: SchemaMigrationPlan {
    public static var schemas: [any VersionedSchema.Type] {
        return [TC2AttestationStoreSchema_v1.self]
    }

    public static var stages: [MigrationStage] {
        return []
    }

    public enum TC2AttestationStoreSchema_v1: VersionedSchema {
        public static var models: [any PersistentModel.Type] {
            [TC2ParamsStoreEntry.self, TCNodeStoreEntry.self]
        }

        public static var versionIdentifier: Schema.Version {
            Schema.Version(1, 0, 0)
        }

        @Model
        public final class TC2ParamsStoreEntry: Hashable, Identifiable {
            #Index<TC2ParamsStoreEntry>([\.pipelineKind, \.model, \.adapter, \.batchId])

            var pipelineKind: String
            var model: String
            var adapter: String
            // false implies prewarm attestations
            var isPrefetched: Bool
            var fetchTime: Date
            var uniqueNodeIds: [String] = []
            var batchId: UInt
            // This is a serverRequestID always
            var usedByTrustedRequestWithId: UUID?

            init(
                pipelineKind: String,
                model: String,
                adapter: String,
                isPrefetched: Bool,
                fetchTime: Date,
                uniqueNodeId: String,
                batchId: UInt
            ) {
                self.pipelineKind = pipelineKind
                self.model = model
                self.adapter = adapter
                self.isPrefetched = isPrefetched
                self.fetchTime = fetchTime
                self.uniqueNodeIds = [uniqueNodeId]
                self.batchId = batchId
            }
        }

        @Model
        public final class TCNodeStoreEntry: Hashable, Identifiable {
            #Index<TCNodeStoreEntry>([\.attestationExpiry], [\.uniqueNodeIdentifier, \.attestationExpiry])

            // CloudAttestation's identifier
            @Attribute(.unique) var uniqueNodeIdentifier: String
            // ROPES returned identifier
            var ropesNodeIdentifier: String
            var attestationBundle: Data
            var attestationExpiry: Date
            var publicKey: Data
            var cloudOSVersion: String
            var cloudOSReleaseType: String
            var cellID: String
            var ensembleID: String?

            init(
                uniqueNodeIdentifier: String,
                ropesNodeIdentifier: String,
                attestationBundle: Data,
                attestationExpiry: Date,
                publicKey: Data,
                cloudOSVersion: String,
                cloudOSReleaseType: String,
                cellID: String,
                ensembleID: String?
            ) {
                self.uniqueNodeIdentifier = uniqueNodeIdentifier
                self.ropesNodeIdentifier = ropesNodeIdentifier
                self.attestationBundle = attestationBundle
                self.attestationExpiry = attestationExpiry
                self.publicKey = publicKey
                self.cloudOSVersion = cloudOSVersion
                self.cloudOSReleaseType = cloudOSReleaseType
                self.cellID = cellID
                self.ensembleID = ensembleID
            }
        }
    }
}

@ModelActor
final actor TC2AttestationStore: TC2AttestationStoreProtocol, Sendable {
    init?(environment: TC2Environment, dir: URL) {
        let storesDir = Self.storesDir(rootDir: dir, environment: environment)

        let storeFileURL = storesDir.appendingPathComponent("attestation_store_v0.2.sqlite", isDirectory: false)
        logger.log("\(#function): attestation store path: \(storeFileURL)")
        let configuration = ModelConfiguration(url: storeFileURL)
        guard let currentVersionedSchema = TC2AttestationStoreMigrationPlan.schemas.last else {
            logger.error("failed to init attestation store, missing schema")
            return nil
        }
        let schema = Schema(versionedSchema: currentVersionedSchema)
        do {
            let container = try ModelContainer(for: schema, migrationPlan: TC2AttestationStoreMigrationPlan.self, configurations: configuration)
            self.init(modelContainer: container)
        } catch {
            logger.error("failed to init attestation store, error=\(error)")
            return nil
        }
    }

    private static func storesDir(rootDir: URL, environment: TC2Environment) -> URL {
        let dirName = "Stores_\(environment.name)"
        return rootDir.appendingPathComponent(dirName, isDirectory: true)
    }

    /// This will only be ever called for a brand new Attestation that doesn't have an entry in the TC2NodeStore
    func saveValidatedAttestation(
        _ validatedAttestation: ValidatedAttestation,
        for parameters: TC2RequestParameters,
        prefetched: Bool,
        batch: UInt,
        fetchTime: Date
    ) -> Bool {
        logger.log("\(#function): adding entry for node: \(validatedAttestation.attestation.nodeID) batch: \(batch) prefetched: \(prefetched) fetchTime: \(fetchTime)")

        guard let uniqueNodeIdentifier = validatedAttestation.uniqueNodeIdentifier else {
            logger.error("missing validatedAttestation.uniqueNodeIdentifier")
            return false
        }
        guard let validatedCellID = validatedAttestation.validatedCellID else {
            logger.error("missing validatedAttestation.validatedCellID")
            return false
        }
        guard let bundle = validatedAttestation.attestation.attestationBundle else {
            logger.error("missing validatedAttestation.attestation.attestationBundle")
            return false
        }

        // First create an entry in the TC2NodeStore for this attestation
        let newNodeEntry = TC2NodeStoreEntry(
            uniqueNodeIdentifier: uniqueNodeIdentifier,
            ropesNodeIdentifier: validatedAttestation.attestation.nodeID,
            attestationBundle: bundle,
            attestationExpiry: validatedAttestation.attestationExpiry,
            publicKey: validatedAttestation.publicKey,
            cloudOSVersion: validatedAttestation.attestation.cloudOSVersion,
            cloudOSReleaseType: validatedAttestation.attestation.cloudOSReleaseType,
            cellID: validatedCellID,
            ensembleID: validatedAttestation.attestation.ensembleID
        )
        modelContext.insert(newNodeEntry)

        // Then link the parameters to use this Node
        if let params = fetchParamsEntry(parameters: parameters, batchId: batch) {
            if !params.uniqueNodeIds.contains(uniqueNodeIdentifier) {
                params.uniqueNodeIds.append(uniqueNodeIdentifier)
            }
        } else {
            // Create a new tracking entry for this set of parameters
            logger.log("\(#function): Linking \(uniqueNodeIdentifier) to ...")
            createNewParamsEntry(
                parameters: parameters,
                withNode: uniqueNodeIdentifier,
                isPrefetched: prefetched,
                batchId: batch,
                time: fetchTime)
        }

        do {
            try modelContext.save()
            return true
        } catch {
            logger.error("failed to insert entry: \(error)")
            return false
        }
    }

    func getAllNodesAndAttestations() -> [String: ValidatedAttestation] {
        do {
            // A validated attestation is created for every parameter we have fetched so far
            var attestations: [String: ValidatedAttestation] = [:]
            let prefetchEntries = try modelContext.fetch(FetchDescriptor<TC2ParamsStoreEntry>())
            for prefetchEntry in prefetchEntries {
                let uniqueNodeIds = prefetchEntry.uniqueNodeIds
                let queryNodePredicate = #Predicate<TC2NodeStoreEntry> { entry in
                    uniqueNodeIds.contains(entry.uniqueNodeIdentifier)
                }
                do {
                    let nodes = try modelContext.fetch(FetchDescriptor(predicate: queryNodePredicate))
                    for node in nodes {
                        attestations[node.ropesNodeIdentifier] = .init(
                            entry: node
                        )
                    }
                } catch {
                    // It is possible that the parameter set can be tracking nodes that no longer exist
                    logger.error("failed to query attestations error: \(error)")
                }
            }
            return attestations
        } catch {
            logger.error("failed to query attestations error: \(error)")
            return [:]
        }
    }

    func getAttestationsForRequest(
        parameters: TC2RequestParameters,
        serverRequestID: UUID,
        maxAttestations: Int
    ) -> [String: ValidatedAttestation] {
        logger.log("\(#function) id: \(serverRequestID)")

        let today = Date()
        let (pipelineKind, model, adapter) = TC2PrefetchParameters().prefetchStoreKeys(prefetchParameters: parameters)

        let prefetchPredicate = #Predicate<TC2ParamsStoreEntry> { entry in
            entry.pipelineKind == pipelineKind && entry.model == model && entry.adapter == adapter && entry.usedByTrustedRequestWithId == nil
        }

        do {
            // A validated attestation is created for every parameter we have fetched so far
            var attestations: [String: ValidatedAttestation] = [:]
            let sortByFetchTime = SortDescriptor(\TC2ParamsStoreEntry.fetchTime, order: .forward)
            let prefetchDescriptor = FetchDescriptor(predicate: prefetchPredicate, sortBy: [sortByFetchTime])
            let prefetchEntries = try modelContext.fetch(prefetchDescriptor)

            outerLoop: for batchToUse in prefetchEntries {
                var count = 0
                let uniqueNodeIds = batchToUse.uniqueNodeIds

                let queryNodePredicate = #Predicate<TC2NodeStoreEntry> { entry in
                    uniqueNodeIds.contains(entry.uniqueNodeIdentifier) && entry.attestationExpiry >= today
                }
                do {
                    let nodes = try modelContext.fetch(FetchDescriptor(predicate: queryNodePredicate))
                    innerLoop: for node in nodes {
                        attestations[node.ropesNodeIdentifier] = .init(
                            entry: node
                        )
                        count += 1
                        if count >= maxAttestations {
                            break innerLoop
                        }
                    }
                    if !attestations.isEmpty {
                        batchToUse.usedByTrustedRequestWithId = serverRequestID
                        logger.log("getAttestationsForRequest \(serverRequestID) returned batch: \(batchToUse.batchId) nodes count: \(attestations.count)")
                        if modelContext.hasChanges {
                            try modelContext.save()
                        }
                        break
                    }
                } catch {
                    logger.error("failed to query nodes, error: \(error)")
                }
            }

            return attestations
        } catch {
            logger.error("failed to query unexpired attestations: \(error)")
            return [:]
        }
    }

    public func deleteEntriesWithExpiredAttestationBundles() {
        logger.log("\(#function)")

        // This will just delete the node entries from the NodeStore
        // Parameter store may have stale entries, but that should be ok since there will be no underlying node
        let today = Date()

        let queryPredicate = #Predicate<TC2NodeStoreEntry> { entry in
            today > entry.attestationExpiry
        }

        do {
            try modelContext.delete(model: TC2NodeStoreEntry.self, where: queryPredicate)
        } catch {
            logger.error("failed to delete expired attestations: \(error)")
        }
    }

    public func deleteEntries(
        withParameters parameters: TC2RequestParameters,
        batchId: UInt
    ) {
        let (pipelineKind, model, adapter) = TC2PrefetchParameters().prefetchStoreKeys(prefetchParameters: parameters)
        logger.log("\(#function): pipelineKind: \(pipelineKind), model: \(model), adapter: \(adapter), batchId: \(batchId)")

        let queryPredicate = #Predicate<TC2ParamsStoreEntry> { entry in
            entry.pipelineKind == pipelineKind && entry.model == model && entry.adapter == adapter && entry.batchId == batchId
        }

        do {
            try modelContext.delete(model: TC2ParamsStoreEntry.self, where: queryPredicate)
        } catch {
            logger.error("failed to delete entries: \(error)")
        }
    }

    public func deleteAllAttestationStoreEntries() {
        logger.log("\(#function)")

        do {
            try modelContext.delete(model: TC2NodeStoreEntry.self)
            try modelContext.delete(model: TC2ParamsStoreEntry.self)
        } catch {
            logger.error("failed to delete all entries: \(error)")
        }
    }

    /// Delete a node entry by looking up the ROPES provided identifier
    /// This is called in the invoke path where ROPES may tell the client that a few attestations sent by the client are unusable
    public func deleteEntryForNode(nodeIdentifier: String) -> Bool {
        logger.log("\(#function): \(nodeIdentifier)")

        let queryPredicate = #Predicate<TC2NodeStoreEntry> { entry in
            entry.ropesNodeIdentifier == nodeIdentifier
        }

        do {
            try modelContext.delete(model: TC2NodeStoreEntry.self, where: queryPredicate)
        } catch {
            logger.error("failed to delete entry for node with ropes identifier: \(nodeIdentifier) with error: \(error)")
            return false
        }

        return true
    }

    func nodeExists(
        withUniqueIdentifier uniqueIdentifier: String
    ) -> Bool {
        logger.log("\(#function): checking if \(uniqueIdentifier) node exists")

        // Return true if we have this node (unexpired) at all in our NodeStore to ensure we save on validation effort
        let today = Date()

        let queryPredicate = #Predicate<TC2NodeStoreEntry> { entry in
            entry.uniqueNodeIdentifier == uniqueIdentifier && entry.attestationExpiry >= today
        }

        do {
            return try modelContext.fetchCount(FetchDescriptor(predicate: queryPredicate)) > 0
        } catch {
            logger.error("failed to query nodes: \(error)")
        }

        return false
    }

    /// Checks to see if node with uniqueIdentifier is tracked in a given batch and for a parameters set, if not, it will add the tracking
    /// Returns true if parameters set already tracks this node in this batch
    /// This is to ensure that we calculate duplicates in a batch and not across batches
    func trackNodeForParameters(
        forParameters parameters: TC2RequestParameters,
        withUniqueIdentifier uniqueIdentifier: String,
        prefetched: Bool,
        batchID: UInt,
        fetchTime: Date
    ) -> Bool {
        logger.log("\(#function): checking if \(uniqueIdentifier) node tracks params")

        do {
            // let's ensure that the parameters cache is tracking this entry
            if nodeExistsInBatch(parameters: parameters, uniqueIdentifer: uniqueIdentifier, batchID: batchID) {
                return true
            } else {
                if let params = fetchParamsEntry(parameters: parameters, batchId: batchID) {
                    params.uniqueNodeIds.append(uniqueIdentifier)
                } else {
                    // Create a new tracking entry for this set of parameters and batch
                    logger.log("\(#function): Linking \(uniqueIdentifier) to ...")
                    createNewParamsEntry(
                        parameters: parameters,
                        withNode: uniqueIdentifier,
                        isPrefetched: prefetched,
                        batchId: batchID,
                        time: fetchTime
                    )
                }
            }
            try modelContext.save()
        } catch {
            logger.error("failed to query nodes: \(error)")
        }

        return false
    }

    /// Fetches an entry for a particular parameter set - if it exists in the store
    private func fetchParamsEntry(parameters: TC2RequestParameters, batchId: UInt) -> TC2ParamsStoreEntry? {
        let (pipelineKind, model, adapter) = TC2PrefetchParameters().prefetchStoreKeys(prefetchParameters: parameters)
        logger.log("\(#function): \(pipelineKind) \(model) \(adapter)")

        let queryPredicate = #Predicate<TC2ParamsStoreEntry> { entry in
            entry.pipelineKind == pipelineKind && entry.model == model && entry.adapter == adapter && entry.batchId == batchId
        }

        // Uniqueness attribute cannot be a combination of keys in SwiftData
        // Because of that, we will need to ensure that only one set of parameters are tracked per batch fetched

        // See WWDC Video - This is possible with #Unique
        do {
            let prefetchEntries = try modelContext.fetch(FetchDescriptor(predicate: queryPredicate))
            return prefetchEntries.first
        } catch {
            logger.error("failed to query entries: \(error)")
            return nil
        }
    }

    /// Create a new entry in the parameters table to track a set of previously unknown parameters
    private func createNewParamsEntry(parameters: TC2RequestParameters, withNode: String, isPrefetched: Bool, batchId: UInt, time: Date) {
        // Create a new tracking entry for this set of parameters
        let (pipelineKind, model, adapter) = TC2PrefetchParameters().prefetchStoreKeys(prefetchParameters: parameters)
        logger.log("\(#function): \(pipelineKind) \(model) \(adapter)")

        let newPrefetchEntry = TC2ParamsStoreEntry(
            pipelineKind: pipelineKind,
            model: model,
            adapter: adapter,
            isPrefetched: isPrefetched,
            fetchTime: time,
            uniqueNodeId: withNode,
            batchId: batchId
        )
        modelContext.insert(newPrefetchEntry)
    }

    private func nodeExistsInBatch(parameters: TC2RequestParameters, uniqueIdentifer: String, batchID: UInt) -> Bool {
        let (pipelineKind, model, adapter) = TC2PrefetchParameters().prefetchStoreKeys(prefetchParameters: parameters)
        logger.log("\(#function): \(pipelineKind) \(model) \(adapter)")

        let queryPredicate = #Predicate<TC2ParamsStoreEntry> { entry in
            entry.pipelineKind == pipelineKind && entry.model == model && entry.adapter == adapter && entry.batchId == batchID
        }

        do {
            let prefetchEntry = try modelContext.fetch(FetchDescriptor(predicate: queryPredicate))
            if let prefetchEntry = prefetchEntry.first {
                if prefetchEntry.uniqueNodeIds.contains(uniqueIdentifer) {
                    return true
                }
            }
        } catch {
            logger.error("failed to query entries: \(error)")
        }

        return false
    }

    func attestationsExist(forParameters: TC2RequestParameters, clientCacheSize: Int, fetchTime: Date) -> Bool {
        let (pipelineKind, model, adapter) = TC2PrefetchParameters().prefetchStoreKeys(prefetchParameters: forParameters)
        logger.log("\(#function): pipeline: \(pipelineKind) \(model) \(adapter) clientCacheSize: \(clientCacheSize), fetchTime: \(fetchTime)")

        let queryPredicate = #Predicate<TC2ParamsStoreEntry> { entry in
            entry.pipelineKind == pipelineKind && entry.model == model && entry.adapter == adapter && entry.fetchTime >= fetchTime && entry.usedByTrustedRequestWithId == nil
        }

        do {
            let entries = try modelContext.fetch(FetchDescriptor(predicate: queryPredicate))
            let nodes = entries.flatMap(\.uniqueNodeIds)
            if nodes.count >= clientCacheSize {
                return true
            }
        } catch {
            logger.error("failed to fetch entries from prefetch store: \(error)")
        }

        return false
    }

    // Why is this doing a fetch only to do a batch delete after?
    // suggestion, just do the batch delete, if you need metrics, do a fetchCount
    func deleteAttestationsUsedByTrustedRequest(
        serverRequestID: UUID
    ) -> UInt {
        logger.log("deleteAttestationsUsedForTrustedRequest: \(serverRequestID)")

        let queryPredicate = #Predicate<TC2ParamsStoreEntry> { entry in
            entry.usedByTrustedRequestWithId == serverRequestID
        }

        var deletedBatch: UInt = 0
        do {
            let prefetchEntries = try modelContext.fetch(FetchDescriptor(predicate: queryPredicate))
            if let prefetchEntry = prefetchEntries.first {
                deletedBatch = prefetchEntry.batchId
                logger.log("deleting batch: \(deletedBatch) used by request: \(serverRequestID)")
                try modelContext.delete(model: TC2ParamsStoreEntry.self, where: queryPredicate)
            }
        } catch {
            logger.error("failed to delete entries: \(error)")
        }

        return deletedBatch
    }

    func getAttestationBundlesUsedByTrustedRequest(
        serverRequestID: UUID
    ) -> [String: Data] {
        logger.log("getAttestationBundlesUsedByTrustedRequest: \(serverRequestID)")
        let queryPredicate = #Predicate<TC2ParamsStoreEntry> { entry in
            entry.usedByTrustedRequestWithId == serverRequestID
        }

        var bundles: [String: Data] = [:]
        let today = Date()
        do {
            let prefetchEntries = try modelContext.fetch(FetchDescriptor(predicate: queryPredicate))
            if let prefetchEntry = prefetchEntries.first {
                let uniqueNodeIds = prefetchEntry.uniqueNodeIds
                let queryNodePredicate = #Predicate<TC2NodeStoreEntry> { entry in
                    uniqueNodeIds.contains(entry.uniqueNodeIdentifier) && entry.attestationExpiry >= today
                }
                do {
                    let nodes = try modelContext.fetch(FetchDescriptor(predicate: queryNodePredicate))
                    for node in nodes {
                        bundles[node.ropesNodeIdentifier] = node.attestationBundle
                    }
                } catch {
                    logger.error("failed to query attestations error: \(error)")
                }
            }
        } catch {
            logger.error("failed to delete entries: \(error)")
        }

        return bundles
    }
}

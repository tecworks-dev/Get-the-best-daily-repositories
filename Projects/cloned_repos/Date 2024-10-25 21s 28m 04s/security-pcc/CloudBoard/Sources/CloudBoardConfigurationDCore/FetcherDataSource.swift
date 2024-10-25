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

// Copyright © 2023 Apple. All rights reserved.

import CloudBoardMetrics
import ConfigurationServiceClient
import Foundation
import os

/// A result from calling `FetcherDataSource.fetchLatest(currentRevisionIdentifier:)`.
enum FetchLatestResult: Hashable {
    /// The provided revision identifier is up to date.
    case upToDate

    /// A new revision of the configuration package is available and is provided.
    case newAvailable(NodeConfigurationPackage)
}

/// A data source for fetching the latest configuration package.
protocol FetcherDataSource {
    /// Fetches the latest configuration package.
    /// - Parameter currentRevisionIdentifier: The current revision identifier.
    /// - Returns: The result of the fetch.
    func fetchLatest(currentRevisionIdentifier: String?) async throws -> FetchLatestResult
}

extension FetchLatestResult: CustomStringConvertible {
    var description: String {
        switch self {
        case .upToDate:
            return "upToDate"
        case .newAvailable(let nodeConfigurationPackage):
            return "newAvailable(\(nodeConfigurationPackage))"
        }
    }
}

/// A data source that fetches the latest configuration package from an upstream service.
struct UpstreamFetcherDataSource {
    static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "UpstreamFetcherDataSource"
    )

    /// The upstream service.
    var upstream: any ConfigurationServiceClient.ServiceProtocol

    /// The release info used when querying the upstream service.
    var releaseInfo: ReleaseInfo

    /// The metrics system to use.
    var metrics: MetricsSystem

    /// An error thrown by the upstream data source.
    private enum DataSourceError: Swift.Error, LocalizedError, CustomStringConvertible {
        /// The release info is not found on the upstream service.
        case releaseNotFound(ReleaseInfo)

        var description: String {
            switch self {
            case .releaseNotFound(let releaseInfo):
                return "Release not found: \(releaseInfo)"
            }
        }
    }
}

extension UpstreamFetcherDataSource: FetcherDataSource {
    func fetchLatest(currentRevisionIdentifier: String?) async throws -> FetchLatestResult {
        self.metrics.emit(Metrics.Upstream.TotalRequestsCounter(action: .increment))
        let startInstant = ContinuousClock.now
        let response: ConfigurationPackageResponse
        do {
            response = try await self.upstream.fetchLatestConfigurationPackage(
                projectName: self.releaseInfo.project,
                environmentName: self.releaseInfo.environment,
                releaseIdentifier: self.releaseInfo.release,
                instanceIdentifier: self.releaseInfo.instance,
                currentRevisionIdentifier: currentRevisionIdentifier
            )
        } catch {
            self.metrics.emit(Metrics.Upstream.FailureCounter(action: .increment))
            throw error
        }
        let configurationPackage: RawConfigurationPackage
        switch response {
        case .notFound:
            Self.logger.error("Configured release info not found on the upstream service.")
            self.metrics.emit(Metrics.Upstream.FailureCounter(action: .increment))
            throw DataSourceError.releaseNotFound(self.releaseInfo)
        case .alreadyUpToDate:
            Self.logger
                .info("Already on the up-to-date revision \(currentRevisionIdentifier!, privacy: .public), no action.")
            self.metrics.emit(Metrics.Upstream.AnySuccessCounter(action: .increment))
            self.metrics.emit(Metrics.Upstream.SuccessUpToDateCounter(action: .increment))
            self.metrics.emit(Metrics.Upstream.SuccessDurationHistogram(durationSinceStart: startInstant))
            return .upToDate
        case .newAvailable(let newConfigurationPackage):
            let byteCount = newConfigurationPackage.rawData.count
            self.metrics.emit(Metrics.Upstream.ConfigSizeGauge(value: byteCount))
            Self.logger.log(
                "Fetched a new revision: \(newConfigurationPackage.revisionIdentifier, privacy: .public) with a configuration package of size \(byteCount, privacy: .public) bytes."
            )
            configurationPackage = newConfigurationPackage
        }
        let identifier = configurationPackage.revisionIdentifier
        do {
            let serializable = try SerializableNodeConfigurationPackage(
                decodingJSONData: configurationPackage.rawData
            )
            let configurationPackage = NodeConfigurationPackage(
                revisionIdentifier: identifier,
                serializable: serializable
            )
            Self.logger.log(
                "Successfully downloaded and parsed a new configuration package: \(configurationPackage, privacy: .public)"
            )
            self.metrics.emit(Metrics.Upstream.AnySuccessCounter(action: .increment))
            self.metrics.emit(Metrics.Upstream.SuccessNewConfigCounter(action: .increment))
            self.metrics.emit(Metrics.Upstream.SuccessDurationHistogram(durationSinceStart: startInstant))
            return .newAvailable(configurationPackage)
        } catch {
            Self.logger
                .error(
                    "Failed to parse the downloaded configuration bundle for revision: \(identifier, privacy: .public)."
                )
            self.metrics.emit(Metrics.Upstream.FailureCounter(action: .increment))
            throw error
        }
    }
}

/// A best-effort cache.
///
/// If the cache hits an underlying issue, it just resets itself.
///
/// Use logging and metrics to report issues that need to be investigated, but issues with the cache
/// never fail the critical path of the fetch, that's why none of the methods are throwing.
protocol FetcherCacheProtocol {
    /// Load the cached value.
    ///
    /// If the value is not available, it returns `nil`.
    func load() async -> NodeConfigurationPackage?

    /// Store the value.
    ///
    /// Overwrites any existing cached value.
    func store(_ value: NodeConfigurationPackage) async
}

/// A data source that caches the result of the upstream data source.
///
/// The cached value is used when the upstream data source returns an error.
struct CachingFetcherDataSource {
    static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "CachingFetcherDataSource"
    )

    /// The underlying cache.
    let cache: FetcherCacheProtocol

    /// The upstream data source that is tried first.
    let upstream: any FetcherDataSource
}

extension CachingFetcherDataSource: FetcherDataSource {
    func fetchLatest(currentRevisionIdentifier: String?) async throws -> FetchLatestResult {
        do {
            let value = try await upstream.fetchLatest(currentRevisionIdentifier: currentRevisionIdentifier)
            Self.logger.debug("The upstream data source succeeded.")
            switch value {
            case .upToDate:
                return .upToDate
            case .newAvailable(let configuration):
                await self.cache.store(configuration)
                return .newAvailable(configuration)
            }
        } catch {
            Self.logger.warning("The upstream data source failed.")
            if let cachedValue = await cache.load() {
                Self.logger.info("Using a cached value.")
                let value: FetchLatestResult
                if cachedValue.revisionIdentifier == currentRevisionIdentifier {
                    value = .upToDate
                    Self.logger.debug("The cached value is up-to-date with the requested revision.")
                } else {
                    value = .newAvailable(cachedValue)
                    Self.logger.debug("The cached value is different to the requested revision.")
                }
                return value
            } else {
                Self.logger.warning("No cached value found, rethrowing the upstream error.")
                throw error
            }
        }
    }
}

/// A cache that stores the configuration in a file on disk.
actor OnDiskFetcherCache {
    static let logger: Logger = .init(
        subsystem: "com.apple.cloudos.cloudboard",
        category: "OnDiskFetcherCache"
    )

    /// The directory URL of the cache.
    nonisolated let cacheDirectory: URL

    /// The file URL of the cache JSON file.
    nonisolated let cacheFileURL: URL

    /// The metrics system to use.
    let metrics: MetricsSystem

    /// The file manager used for file operations.
    private let fileManager: FileManager = .init()

    /// The JSON keys used to persist the cache.
    private enum Keys: String {
        case revisionIdentifier
        case configuration
    }

    /// Creates a new cache.
    /// - Parameters:
    ///   - cacheDirectory: The directory URL of the cache.
    ///   - metrics: The metrics system to use.
    /// - Throws: If the directory couldn't be created.
    init(cacheDirectory: URL, metrics: MetricsSystem) throws {
        self.cacheDirectory = cacheDirectory
        try self.fileManager.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        self.cacheFileURL = cacheDirectory.appending(component: "fetcher-cache.json")
        self.metrics = metrics
    }

    /// Resets the cache by removing the underlying cached file.
    private func reset() {
        do {
            self.metrics.emit(Metrics.Cache.CacheResetCounter(action: .increment))
            self.metrics.emit(Metrics.Cache.CacheSizeGauge(value: 0))
            try self.fileManager.removeItem(at: self.cacheFileURL)
        } catch {
            Self.logger.error(
                "The cache failed to delete a malformed file, error: \(String(unredacted: error), privacy: .public). The cache will probably not recover now and will keep failing."
            )
        }
    }
}

extension OnDiskFetcherCache: FetcherCacheProtocol {
    func load() async -> NodeConfigurationPackage? {
        let data: Data
        do {
            data = try Data(contentsOf: self.cacheFileURL, options: .uncached)
            Self.logger.debug("Cache file data loaded.")
        } catch let error as NSError
            where error.domain == NSCocoaErrorDomain && error.code == NSFileReadNoSuchFileError {
            Self.logger.debug("Cache miss.")
            self.metrics.emit(Metrics.Cache.CacheMissCounter(action: .increment))
            return nil
        } catch {
            Self.logger.error(
                "The cache is malformed, resetting. Error: \(String(unredacted: error), privacy: .public)"
            )
            self.reset()
            return nil
        }
        do {
            let onDiskConfiguration = try OnDiskCachedConfiguration(decodingJSONData: data)
            self.metrics.emit(Metrics.Cache.CacheHitCounter(action: .increment))
            return NodeConfigurationPackage(
                revisionIdentifier: onDiskConfiguration.revisionIdentifier,
                serializable: onDiskConfiguration.configuration
            )
        } catch {
            Self.logger.error(
                "The cache is malformed, resetting. Error: \(String(unredacted: error), privacy: .public)"
            )
            self.reset()
            return nil
        }
    }

    func store(_ value: NodeConfigurationPackage) async {
        do {
            self.metrics.emit(Metrics.Cache.CacheStoreCounter(action: .increment))
            let onDiskConfiguration = try OnDiskCachedConfiguration(value).encodedAsJSONData()
            self.metrics.emit(Metrics.Cache.CacheSizeGauge(value: onDiskConfiguration.count))
            try onDiskConfiguration.write(to: self.cacheFileURL)
            Self.logger.debug("Cache file data for revision \(value.revisionIdentifier, privacy: .public) saved.")
        } catch {
            Self.logger.error(
                "Failed to store the configuration to disk. Error: \(String(unredacted: error), privacy: .public)"
            )
            self.reset()
        }
    }
}

/// The serialized representation of a cached configuration package.
private struct OnDiskCachedConfiguration {
    /// The revision identifier for the configuration package.
    let revisionIdentifier: String

    /// The serialized representation of the configuration package.
    let configuration: SerializableNodeConfigurationPackage
}

extension OnDiskCachedConfiguration {
    init(_ configuration: NodeConfigurationPackage) {
        self.init(
            revisionIdentifier: configuration.revisionIdentifier,
            configuration: .init(domains: configuration.domains)
        )
    }
}

extension NodeConfigurationPackage {
    fileprivate init(_ configuration: OnDiskCachedConfiguration) {
        self.init(
            revisionIdentifier: configuration.revisionIdentifier,
            serializable: configuration.configuration
        )
    }
}

extension OnDiskCachedConfiguration {
    enum Keys: String {
        case revisionIdentifier
        case configuration
    }

    func encodedAsJSONObject() -> [String: Any] {
        [
            Keys.revisionIdentifier.rawValue: self.revisionIdentifier,
            Keys.configuration.rawValue: self.configuration.encodedAsJSONObject(),
        ]
    }

    init(decodingJSONObject object: Any) throws {
        enum DecodingError: Error, LocalizedError, CustomStringConvertible {
            case invalidTopLevelType
            case missingKey(String)
            case invalidValueForKey(String)
            case error(revisionIdentifier: String, underlyingError: Error)

            var description: String {
                switch self {
                case .invalidTopLevelType:
                    return "The on disk cached configuration package has incorrect structure."
                case .missingKey(let key):
                    return "The on disk cached configuration package is missing the required key: \(key)."
                case .invalidValueForKey(let key):
                    return "The on disk cached configuration package has an invalid value for the key: \(key)."
                case .error(let revisionIdentifier, let underlyingError):
                    return "An error occurred while parsing the on disk cache of revision \(revisionIdentifier): \(underlyingError)"
                }
            }

            var errorDescription: String? {
                self.description
            }
        }
        guard let topLevel = object as? [String: Any] else {
            throw DecodingError.invalidTopLevelType
        }
        guard let revisionIdentifierUncheckedValue = topLevel[Keys.revisionIdentifier.rawValue] else {
            throw DecodingError.missingKey(Keys.revisionIdentifier.rawValue)
        }
        guard let revisionIdentifier = revisionIdentifierUncheckedValue as? String else {
            throw DecodingError.invalidValueForKey(Keys.revisionIdentifier.rawValue)
        }
        do {
            guard let configurationUncheckedValue = topLevel[Keys.configuration.rawValue] else {
                throw DecodingError.missingKey(Keys.configuration.rawValue)
            }
            guard let configurationJSONObject = configurationUncheckedValue as? [String: Any] else {
                throw DecodingError.invalidValueForKey(Keys.configuration.rawValue)
            }
            let configuration = try SerializableNodeConfigurationPackage(decodingJSONObject: configurationJSONObject)
            self.init(
                revisionIdentifier: revisionIdentifier,
                configuration: configuration
            )
        } catch {
            throw DecodingError.error(revisionIdentifier: revisionIdentifier, underlyingError: error)
        }
    }

    func encodedAsJSONData() throws -> Data {
        try JSONSerialization.data(
            withJSONObject: self.encodedAsJSONObject(),
            options: [.prettyPrinted, .sortedKeys]
        )
    }

    init(decodingJSONData data: Data) throws {
        enum DecodingError: Error, LocalizedError, CustomStringConvertible {
            case deserializationError(Error)

            var description: String {
                switch self {
                case .deserializationError(let error):
                    return "Failed to deserialize a configuration package from disk with error: \(error)"
                }
            }

            var errorDescription: String? {
                self.description
            }
        }
        do {
            let value = try JSONSerialization.jsonObject(with: data)
            try self.init(decodingJSONObject: value)
        } catch {
            throw DecodingError.deserializationError(error)
        }
    }
}

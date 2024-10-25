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
//  CryptexCache.swift
//	DarwinInit
//

import Foundation
import System
import CryptoKit
import DarwinPrivate.os.variant

extension DInitCryptexConfig {
	var cacheEntryName: String {
		DInitSHA256Digest(SHA256.hash(data: url.dataRepresentation)).description
	}
}

class CryptexCache {
	static let logger = Logger.cryptexcache
	
	enum Error: Swift.Error {
		case failedToConvertDigestToSymmetricKey(DInitSHA256Digest)
		case failedToEnumerateCacheContents
		case evictionFailure
		case validationFailure
		case extractionFailure
	}

	protocol Delegate {
		// Handlers should return the path to the directory of the extracted cryptex.
		// NOTE: The "cacheInsertionHandler" will move (NOT copy) the file into the cache.
		func handleCacheMiss(config: DInitCryptexConfig, cacheInsertionHandler: (FilePath) throws -> Void, tempDir: FilePath) async throws -> FilePath
		func handleCacheHit(config: DInitCryptexConfig, cachedFile: FilePath, tempDir: FilePath) async throws -> FilePath
	}
	
	enum CacheMissError: Swift.Error {
		case downloadFailed
		case failedToCreateCacheEntry(DInitCryptexConfig.DownloadedCryptex)
	}

	enum FetchResult {
		case nonCacheable // The cryptex cache will not handle (download or extract) non-cacheable cryptexes.
		case found(DInitCryptexConfig.ExtractedCryptex)
		case cachingFailure(DInitCryptexConfig.DownloadedCryptex)
	}
	
	private class LoggingDelegate: Delegate {
		let wrappedDelegate: Delegate
		
		init(wrapping delegate: Delegate) {
			self.wrappedDelegate = delegate
		}
		
		func handleCacheMiss(config: DInitCryptexConfig, cacheInsertionHandler: (FilePath) throws -> Void, tempDir: FilePath) async throws -> FilePath {
			CryptexCache.logger.log("[\(config.url)] Cache miss")
			return try await self.wrappedDelegate.handleCacheMiss(config: config, cacheInsertionHandler: cacheInsertionHandler, tempDir: tempDir)
		}
		
		func handleCacheHit(config: DInitCryptexConfig, cachedFile: FilePath, tempDir: FilePath) async throws -> FilePath {
			return try await self.wrappedDelegate.handleCacheHit(config: config, cachedFile: cachedFile, tempDir: tempDir)
		}
	}
	
	class DefaultDelegate: Delegate {
		func handleCacheMiss(config: DInitCryptexConfig, cacheInsertionHandler: (FilePath) throws -> Void, tempDir: FilePath) async throws -> FilePath {
			// The cache entries are stored as Apple Encrypted Archives.
			guard let downloadedCryptex = await config.download(to: tempDir) else {
				CryptexCache.logger.error("[\(config.url)] Failed to download")
				throw CacheMissError.downloadFailed
			}

			// Set up path to extract cryptex to
			guard let extractedCryptexPath = LibCryptex.setUpExtractedPath() else {
				throw Error.extractionFailure
			}

			let downloadedCryptexArchiveMagic: FilePath.ArchiveType?

			do {
				downloadedCryptexArchiveMagic = try downloadedCryptex.path.readArchiveMagic()
			} catch {
				logger.error("[\(config.url)] Failed to read archive magic for \(downloadedCryptex.path): \(error)")
				throw Error.extractionFailure
			}

			switch downloadedCryptexArchiveMagic {
			case .AppleEncryptedArchive:
				guard let extractedCryptexPath = await LibCryptex.extractCryptex(
					at: downloadedCryptex.path,
					url: config.url,
					dawToken: config.dawToken,
					wgUsername: config.wgUsername,
					wgToken: config.wgToken,
					altCDN: config.alternateCDNHost,
					retries: config.networkRetryCount,
					aeaDecryptionParams: config.aeaDecryptionParams) else {
					throw Error.extractionFailure
				}

				do {
					// If the file is already an AEA, just move it into the cache
					try cacheInsertionHandler(downloadedCryptex.path)
				} catch {
					// Failed to insert into cache, so the next time will have a cache miss,
					// but extraction succeeded, so we can continue.
					CryptexCache.logger.warning("[\(config.url)] Failed to insert Apple Encrypted Archive into cache: \(error)")
				}
				
				return extractedCryptexPath
			case .UncompressedAppleArchive:
				try downloadedCryptex.path.extractUncompressedAppleArchive(to: extractedCryptexPath)
			default:
				try downloadedCryptex.path.extract(to: extractedCryptexPath)
			}

			do {
				let extractedContents = try extractedCryptexPath.performDeepEnumerationOfFiles()
				logger.info("Extracted \(downloadedCryptex.path.description) with contents: \(extractedContents)")
			} catch {
				logger.error("Failed to enumerate the contents of \(extractedCryptexPath): \(error)")
			}

			// Non-AEA cases must be wrapped in an AEA before inserted into the cache
			guard let downloadedFileName = downloadedCryptex.path.lastComponent else {
				CryptexCache.logger.error("[\(config.url)] Failed to get file name of downloaded file")
					throw CacheMissError.failedToCreateCacheEntry(downloadedCryptex)
			}

			let stagedCacheArchive = tempDir.appending("\(downloadedFileName.stem).aea")

			guard let digest = config.sha256 else {
				CryptexCache.logger.error("[\(config.url)] No digest found when digest is required.")
				throw Error.validationFailure
			}
			
			do {
				let key = try CryptexCache.generateKey(from: digest)

				try downloadedCryptex.path.wrapFileContentsInAppleEncryptedArchive(
					at: stagedCacheArchive,
					using: key,
					permissions: [.ownerReadWrite],
					compression: downloadedCryptexArchiveMagic == .UncompressedAppleArchive)

				try cacheInsertionHandler(stagedCacheArchive)
			} catch {
				// Failed to insert into cache, so the next time will have a cache miss,
				// but extraction succeeded, so we can continue.
				CryptexCache.logger.warning("[\(config.url)] Failed to insert archive into cache: \(error)")
			}

			return extractedCryptexPath
		}
		
		func handleCacheHit(config: DInitCryptexConfig, cachedFile: FilePath, tempDir: FilePath) async throws -> FilePath {
			let finalArchivePath: FilePath
			
			if config.url.scheme == kKnoxURLScheme || config.aeaDecryptionParams != nil {
				finalArchivePath = cachedFile
			} else {
				let innerArchivePath = tempDir.appending("inner")

				guard let digest = config.sha256 else {
					CryptexCache.logger.error("[\(config.url)] No digest found when digest is required.")
					throw Error.validationFailure
				}

				do {
					try cachedFile.extractAppleEncryptedArchive(
						to: innerArchivePath,
						using: CryptexCache.generateKey(from: digest),
						mode: .extractRawBytes(outputFilePermissions: [.ownerReadWrite]))
				} catch {
					// Give an opportunity for a second chance.
					throw Error.validationFailure
				}

				if (try? innerArchivePath.fileExists()) != true {
					// If the path doesn't point to a regular file, bail.
					throw Error.extractionFailure
				}

				do {
					try innerArchivePath.sha256Equals(expectedSHA256: digest)
				} catch {
					do {
						try innerArchivePath.remove()
					} catch {
						CryptexCache.logger.fault("Failed to remove \"\(innerArchivePath)\" after failed validation: \(error)")
						throw error
					}
					throw CryptexCache.Error.validationFailure
				}
				
				finalArchivePath = innerArchivePath
			}

			guard let extractedCryptexPath = await LibCryptex.extractCryptex(
				at: finalArchivePath,
				url: config.url,
				dawToken: config.dawToken,
				wgUsername: config.wgUsername,
				wgToken: config.wgToken,
				altCDN: config.alternateCDNHost,
				retries: config.networkRetryCount,
				aeaDecryptionParams: config.aeaDecryptionParams) else {
				// Validation is done by this function for Knox and non-Knox AEA archives
				if config.url.scheme == kKnoxURLScheme || config.aeaDecryptionParams != nil {
					throw Error.validationFailure
				} else {
					throw Error.extractionFailure
				}
			}

			return extractedCryptexPath
		}
	}

	private let cacheDirectoryPath: FilePath
	private let maxTotalSize: Int
	private let delegate: Delegate

#if os(macOS)
	internal static let defaultCacheDirectoryPath: FilePath = "/System/Volumes/Preboot/DarwinInitCache"
#else
	internal static let defaultCacheDirectoryPath: FilePath = "/private/preboot/DarwinInitCache"
#endif
	
	init?(at cacheDirectoryPath: FilePath, delegate: Delegate = DefaultDelegate(), maxTotalSize: Int = Int.max) {
		self.cacheDirectoryPath = cacheDirectoryPath
		self.maxTotalSize = max(0, maxTotalSize)
		self.delegate = LoggingDelegate(wrapping: delegate)

		do {
			try CryptexCache.createDirectory(at: self.cacheDirectoryPath)
		} catch {
			CryptexCache.logger.fault("Failed to create CryptexCache directory at \(self.cacheDirectoryPath): \(error)")
			return nil
		}
	}

	class func generateKey(from digest: DInitSHA256Digest) throws -> SymmetricKey {
		guard let bytes = digest.description.hexadecimalASCIIBytes else {
			throw CryptexCache.Error.failedToConvertDigestToSymmetricKey(digest)
		}

		return SymmetricKey(data: SHA256.hash(data: Data(bytes)))
	}

	struct CacheItemInfo {
		let totalFileAllocatedSize: Int
		let contentAccessDate: Date
	}

	func fetch<T>(configs: [DInitCryptexConfig], removeCacheEntriesNotInInputSet: Bool = true, _ body: (DInitCryptexConfig, FetchResult) async throws -> T?) async throws -> [T] {
		let annotatedConfigs = configs.map { config in
			let isCacheable = (config.cacheable != false) && ((config.sha256 != nil) || (config.url.scheme == kKnoxURLScheme) || (config.aeaDecryptionParams?.aeaArchiveId != nil))
			let cacheEntryName = config.cacheEntryName
			return (config, isCacheable, cacheEntryName)
		}
		
		let inputSet = Set<String>(annotatedConfigs.compactMap { config, isCacheable, cacheEntryName in
			return (isCacheable) ? (cacheEntryName) : (nil)
		})

		var cacheContents: [String:CacheItemInfo] = [:]
		var cacheContentsTotalSize: Int = 0

		let resourceKeys = Set<URLResourceKey>([
			.isRegularFileKey,
			.totalFileAllocatedSizeKey,
			.contentAccessDateKey])

		guard let directoryEnumerator = FileManager.default.enumerator(
			at: URL(filePath: self.cacheDirectoryPath.description, directoryHint: .isDirectory),
			includingPropertiesForKeys: Array(resourceKeys),
			options: .skipsSubdirectoryDescendants)
		else {
			CryptexCache.logger.error("Failed to create directory enumerator")
			throw CryptexCache.Error.failedToEnumerateCacheContents
		}

		for case let fileSystemItem as URL in directoryEnumerator {
			guard let resourceValues = try? fileSystemItem.resourceValues(forKeys: resourceKeys),
				  let isRegularFile = resourceValues.isRegularFile
			else {
				CryptexCache.logger.fault("Failed to get \"isRegularFile\" for \(fileSystemItem.lastPathComponent)")
				throw CryptexCache.Error.failedToEnumerateCacheContents
			}

			if !isRegularFile {
				CryptexCache.logger.debug("Removing \"\(fileSystemItem.lastPathComponent)\" from cache because it is not a regular file")
				try CryptexCache.removeItem(at: fileSystemItem)
				continue
			}

			guard let totalFileAllocatedSize = resourceValues.totalFileAllocatedSize,
				  let contentAccessDate = resourceValues.contentAccessDate
			else {
				CryptexCache.logger.fault("Failed to get file attributes for \(fileSystemItem.lastPathComponent)")
				throw CryptexCache.Error.failedToEnumerateCacheContents
			}

			CryptexCache.logger.debug("\"\(fileSystemItem.lastPathComponent)\" size: \(totalFileAllocatedSize.formatted(.byteCount(style: .memory)))")

			if removeCacheEntriesNotInInputSet && !inputSet.contains(fileSystemItem.lastPathComponent) {
				CryptexCache.logger.log("Removing \"\(fileSystemItem.lastPathComponent)\" from cache")
				try CryptexCache.removeItem(at: fileSystemItem)
				continue
			}

			let cacheItemInfo = CacheItemInfo(
				totalFileAllocatedSize: totalFileAllocatedSize,
				contentAccessDate: contentAccessDate)
			cacheContents[fileSystemItem.lastPathComponent] = cacheItemInfo

			cacheContentsTotalSize += totalFileAllocatedSize
		}

		let tempDirRoot = self.cacheDirectoryPath.appending("tmp")
		let outputArray: [T] = try await CryptexCache.withTemporaryDirectory(at: tempDirRoot) {
			return await annotatedConfigs.asyncSerialCompactMap { (config, isCacheable, cacheEntryName) in
				if (isCacheable) {
					let cacheFilePath = self.cacheDirectoryPath.appending(cacheEntryName)
					let tempDir = tempDirRoot.appending(cacheEntryName)
					
					CryptexCache.logger.log("[\(config.url)] Fetching from cache")

					return try? await CryptexCache.withTemporaryDirectory(at: tempDir) {
						func cacheInsertionHandler(inputFilePath: FilePath) throws -> Void {
							let resourceKeys = Set<URLResourceKey>([
								.totalFileAllocatedSizeKey,
								.creationDateKey])

							guard let url = URL(filePath: inputFilePath),
								  let resourceValues = try? url.resourceValues(forKeys: resourceKeys),
								  let totalFileAllocatedSize = resourceValues.totalFileAllocatedSize,
								  let creationDate = resourceValues.creationDate
							else {
								CryptexCache.logger.error("Failed to get file attributes for \(inputFilePath.description)")
								throw CryptexCache.Error.failedToEnumerateCacheContents
							}

							CryptexCache.logger.debug("[\(config.url)] Size: \(totalFileAllocatedSize.formatted(.byteCount(style: .memory))) (Current Total Cache Size: \(cacheContentsTotalSize.formatted(.byteCount(style: .memory))))")

							while (cacheContentsTotalSize + totalFileAllocatedSize) > self.maxTotalSize {
								guard let evictionCandidate = cacheContents.min(by: { a, b in a.value.contentAccessDate < b.value.contentAccessDate }) else {
									CryptexCache.logger.error("[\(config.url)] No more cache entries to evict! New archive too big for cache!")
									throw CryptexCache.Error.evictionFailure
								}

								do {
									CryptexCache.logger.info("Evicting \"\(evictionCandidate.key)\" from cache (\(evictionCandidate.value.totalFileAllocatedSize.formatted(.byteCount(style: .memory))))")
									try CryptexCache.removeItem(at: self.cacheDirectoryPath.appending(evictionCandidate.key))
									cacheContentsTotalSize -= evictionCandidate.value.totalFileAllocatedSize
									cacheContents.removeValue(forKey: evictionCandidate.key)
								} catch {
									CryptexCache.logger.error("Failed to evict cache entry \"\(evictionCandidate.key)\"")
									throw error
								}
							}

							CryptexCache.logger.debug("[\(config.url)] Moving \(inputFilePath) into cache entry at \(cacheFilePath)")

							try FileManager.default.moveItem(
								atPath: inputFilePath.description,
								toPath: cacheFilePath.description)

							let cacheItemInfo = CacheItemInfo(
								totalFileAllocatedSize: totalFileAllocatedSize,
								contentAccessDate: creationDate)
							cacheContents[cacheEntryName] = cacheItemInfo

							cacheContentsTotalSize += totalFileAllocatedSize
						}

						let extractedCryptexDir: FilePath

						if cacheContents[cacheEntryName] == nil {
							do {
								extractedCryptexDir = try await self.delegate.handleCacheMiss(
									config: config,
									cacheInsertionHandler: cacheInsertionHandler,
									tempDir: tempDir)
							} catch CacheMissError.failedToCreateCacheEntry(let downloadedFile) {
								CryptexCache.logger.warning("[\(config.url)] Failed to create cache entry; falling back to using downloaded file directly")

								do {
									return try await body(config, FetchResult.cachingFailure(downloadedFile))
								} catch {
									CryptexCache.logger.error("[\(config.url)] Failed to handle caching failure: \(error)")
									return nil
								}
							} catch {
								CryptexCache.logger.error("[\(config.url)] Failed to handle cache miss: \(error)")
								return nil
							}
						} else {
							CryptexCache.logger.log("[\(config.url)] Found in cache")

							do {
								do {
									extractedCryptexDir = try await self.delegate.handleCacheHit(
										config: config,
										cachedFile: cacheFilePath,
										tempDir: tempDir)
								} catch Error.validationFailure {
									CryptexCache.logger.log("[\(config.url)] Cache entry validation failed; triggering cache miss")

									do {
										try cacheFilePath.remove()
									} catch {
										CryptexCache.logger.fault("[\(config.url)] Failed to remove file that failed validation \(cacheFilePath): \(error)")
										throw error
									}

									// Clean slate for the cache miss
									try tempDir.removeAllFilesInDirectory()

									extractedCryptexDir = try await self.delegate.handleCacheMiss(
										config: config,
										cacheInsertionHandler: cacheInsertionHandler,
										tempDir: tempDir)
								}
							} catch {
								CryptexCache.logger.error("[\(config.url)] Failed to handle cache hit: \(error)")
								return nil
							}
						}

						do {
							return try await body(config, FetchResult.found(
								DInitCryptexConfig.ExtractedCryptex(config: config, path: extractedCryptexDir)))
						} catch {
							CryptexCache.logger.error("[\(config.url)] Failed to handle fetch: \(error)")
							return nil
						}
					}
				} else {
					CryptexCache.logger.log("[\(config.url)] Not cacheable")
					do {
						return try await body(config, FetchResult.nonCacheable)
					} catch {
						CryptexCache.logger.error("Failed to handle non-cacheable cryptex: \(error)")
						return nil
					}
				}
			}
		}
		
		return outputArray
	}
	
	static private func removeItem(at url: URL) throws {
		logger.debug("Removing item at \(url)")
		do {
			try FileManager.default.removeItem(at: url)
		} catch {
			CryptexCache.logger.fault("Failed to remove \"\(url)\" from cache: \(error)")
			throw error
		}
	}

	static private func removeItem(at path: FilePath) throws {
		do {
			try path.remove()
		} catch {
			CryptexCache.logger.fault("Failed to remove \"\(path)\" from cache: \(error)")
			throw error
		}
	}

	static private func createDirectory(at path: FilePath) throws {
		try path.createDirectory(permissions: .ownerReadWriteExecute, intermediateDirectories: true)
	}

	static private func withTemporaryDirectory<T>(at path: FilePath, _ body: () async throws -> T) async throws -> T {
		let returnValue: T

		try CryptexCache.createDirectory(at: path)

		do {
			returnValue = try await body()
		} catch {
			let preexistingError = error

			do {
				try CryptexCache.removeItem(at: path)
			} catch {
				CryptexCache.logger.fault("Removal of file system item failed after pre-existing error thrown: \(preexistingError)")
				throw error
			}

			throw preexistingError
		}

		try CryptexCache.removeItem(at: path)

		return returnValue
	}
}

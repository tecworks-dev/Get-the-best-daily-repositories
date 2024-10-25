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
//  Network.swift
//  darwin-init
//

import Foundation
import Network
import os
import System
import SystemConfiguration
import Darwin
import DarwinPrivate
import Darwin.POSIX.sys.socket
import libnarrativecert

enum Network {
    static let logger = Logger.network

    // visible for testing
    static var urlSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        return URLSession(configuration: configuration)
    }()

    private class TimeoutDelegate: NSObject, URLSessionTaskDelegate {
        let timeout: TimeInterval
        
        init(timeout t: Duration) {
            self.timeout = TimeInterval(t.components.seconds) + Double(t.components.attoseconds)/1e18
        }

        func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
            logger.debug("ulrSession.didCreateTask: \(task), setting timeout to \(self.timeout)")
            task._timeoutIntervalForResource = timeout
        }
    }
    
    private class NarrativeURLSessionDelegate: NSObject, URLSessionTaskDelegate {
        public let cred: URLCredential?
        public init(cred: URLCredential?) {
            self.cred = cred
        }

        public func urlSession(
            _ session: URLSession,
            didReceive challenge: URLAuthenticationChallenge,
            completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
        ) {
            if (challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodClientCertificate) {
                completionHandler(.useCredential, cred)
            } else {
                completionHandler(.performDefaultHandling, nil);
            }
        }
    }

    static func download(
        from url: URL,
        to path: FilePath,
        attempts maxAttempts: Int = 1,
        backoff: BackOff = .linear(.seconds(10), offset: .seconds(5))
    ) async throws {
        if !Time.isSynchronized {
            logger.warning("Time is not synced before making network request, continuing")
        }

        logger.log("Downloading from \(url) to \(path)")

        var request = URLRequest(url: url)
        // disable automatic urlsession 'Accept-Encoding: gzip'
        request.addValue("identity", forHTTPHeaderField: "Accept-Encoding")

        // Attempt to fetch Narrative cert from key chain for authenticated CDN downloads if acdc actor identity has been configured
        let cert = NarrativeCert(domain: .acdc, identityType: .actor)
        let credential = cert.getCredential()
        if credential == nil {
            logger.debug("Failed to create URL credential for auth challenge. Narrative identity may not be configured properly.")
        } else {
            logger.debug("Successfully created URL credential for auth challenge")
        }
        let delegate = NarrativeURLSessionDelegate(cred: credential)

        try await retry(count: maxAttempts, backoff: backoff) { attempt in
            let fileURL: URL
            let response: URLResponse

            do {
                (fileURL, response) = try await urlSession.download(for: request, delegate: delegate)
            } catch {
                throw Network.Error.connectionError(error)
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                throw Network.Error.noResponse
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                throw Network.Error.badResponse(httpResponse.statusCode)
            }

            guard let source = FilePath(fileURL) else {
                throw Network.Error.noData
            }

            defer {
                do {
                    try source.remove()
                } catch {
                    logger.error("Failed to remove temp file \(path)")
                }
            }

            do {
                try source.copy(to: path)
            } catch {
                throw Network.Error.dataTransformFailed(error)
            }

        } shouldRetry: { error in
            logger.error("Download attempt \(url) failed: \(error.localizedDescription)")
            guard let error = error as? Network.Error else {
                return false
            }
            return error.shouldRetry
        }
    }
    
    static func downloadItem(
        at url: URL,
        to destinationDirectory: FilePath? = nil,
        attempts: Int = 3,
        backoff: BackOff = .linear(.seconds(10), offset: .seconds(5))
    ) async -> FilePath? {
        guard let destinationDirectory = destinationDirectory ?? FilePath.createTemporaryDirectory() else {
            return nil
        }
        
        let name = if url.lastPathComponent.isEmpty || url.lastPathComponent == "/" {
            url.host() ?? "download"
        } else {
            url.lastPathComponent
        }
        
        let destinationPath = destinationDirectory.appending(name)

        if let localFilePath = FilePath(url) {
            do {
                try localFilePath.copy(to: destinationPath)
            } catch {
                logger.error("Failed to copy contents from \(localFilePath) to \(destinationPath): \(error.localizedDescription)")
                return nil
            }
        } else {
            do {
                try await download(from: url, to: destinationPath, attempts: attempts, backoff: backoff)
            } catch {
                logger.error("Download failed: \(error.localizedDescription)")
                return nil
            }
        }

        return destinationPath
    }

    /// Performs the `request` and asynchronously returns the response.
    ///
    /// - parameter request: The url request to perform.
    /// - parameter attempts: The maximum number of retries to attempt
    /// - parameter timeout: Maximum time allowed for a single attempt
    /// - parameter backoff: The backoff strategy to use when retrying
    ///
    /// - Returns: The contents of the URL specified by the request as a `Data` instance
    private static func perform(
        request: URLRequest,
        attempts maxAttempts: Int = 3,
        timeout: Duration = .seconds(10),
        backoff: BackOff = .linear(.seconds(10), offset: .seconds(5))
    ) async throws -> Data {
        if !Time.isSynchronized {
            logger.warning("Time is not synced before making network request, continuing")
        }
        let id = UUID()
        logger.log("Performing HTTP request \(id) \(request.logDescription)")
       
        let sessionDelegate = TimeoutDelegate(timeout: timeout)
        return try await retry(count: maxAttempts, backoff: backoff) { attempt in
            let data: Data
            let response: URLResponse

            do {
                (data, response) = try await urlSession.data(for: request, delegate: sessionDelegate)
            } catch {
                throw Network.Error.connectionError(error)
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                throw Network.Error.noResponse
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                throw Network.Error.badResponse(httpResponse.statusCode)
            }

            return data

        } shouldRetry: { error in
            logger.error("Request (\(id)) attempt failed: \(error.localizedDescription)")
            guard let error = error as? Network.Error else {
                return false
            }
            return error.shouldRetry
        }
    }

    /// Encodes the request as JSON and uploads it to the URL via the HTTP POST method.
    static func post<Request: Encodable>(
        _ request: Request,
        to url: URL,
        attempts: Int = 3,
        timeout: Duration = .seconds(10),
        backoff: BackOff = .linear(.seconds(10), offset: .seconds(5))
    ) async throws -> Data {
        var urlRequest = URLRequest(url: url)
        urlRequest.httpBody = try JSONEncoder().encode(request)
        urlRequest.httpMethod = "POST"
        return try await perform(request: urlRequest, attempts: attempts, timeout: timeout, backoff: backoff)
    }

    /// Fetches the content of the URL via a HTTP GET request.
    static func get(
        from url: URL,
        additionalHTTPHeaders: [String: String] = [:],
        attempts: Int = 3,
        timeout: Duration = .seconds(10),
        backoff: BackOff = .linear(.seconds(10), offset: .seconds(5))
    ) async throws -> Data {
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"
        urlRequest.addHeaders(additionalHTTPHeaders: additionalHTTPHeaders)
        return try await perform(request: urlRequest, attempts: attempts, timeout: timeout, backoff: backoff)
    }
    
    /// Performs a HTTP PUT request to the given URL.
    static func put(
        to url: URL,
        additionalHTTPHeaders: [String: String] = [:],
        attempts: Int = 3,
        timeout: Duration = .seconds(10),
        backoff: BackOff = .linear(.seconds(10), offset: .seconds(5))
    ) async throws -> Data {
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "PUT"
        urlRequest.addHeaders(additionalHTTPHeaders: additionalHTTPHeaders)
        return try await perform(request: urlRequest, attempts: attempts, timeout: timeout)
    }

    // Helper for getting Mellanox interface bsd name on J236
    private static func getUplinkInterfaceName() throws -> String {
        try retry(count: 5, delay: .seconds(5), backoff: .seconds(10)) { attempt in
            let domain = CFPreferences.Domain(
                applicationId: kUplinkInterfaceAppID as CFString,
                userName: kCFPreferencesAnyUser,
                hostName: kCFPreferencesCurrentHost)
            
            guard let bsdName = CFPreferences.getValue(for: kUplinkInterfaceKey, in: domain) else {
                logger.debug("Reattempting CFPref read of \(kUplinkInterfaceKey)...")
                throw UplinkInterfaceError("Failed to read \(kUplinkInterfaceKey) value")
            }
            
            return bsdName as! String
        }
    }
    
    static func unsetUplinkBandwidthLimit() -> Bool {
        guard setUplinkBandwidthLimit(bandwidthLimit: 0) else {
            logger.error("Failed to reset uplink interface bandwidth")
            return false
        }
        return true
    }
    
    // Configure the bandwidth limit on the Mellanox interface
    static func setUplinkBandwidthLimit(bandwidthLimit: UInt64) -> Bool {
        var bsdName: String
        do {
            try bsdName = getUplinkInterfaceName()
        } catch {
            logger.error("Failed to get uplink interface bsd name: \(error)")
            return false
        }
        logger.info("Configuring bandwidth limit for interface \(bsdName)...")
        
        // Attempt to open socket, retrying if we fail due to interrupt
        var sock:Int32
        let sockStatus = valueOrErrno(retryOnInterrupt: true) {
            socket(AF_INET, SOCK_DGRAM, 0)
        }
        switch sockStatus {
        case .success(let sockVal):
            sock = sockVal
            logger.info("Opened socket: \(sock)")
            break
        case .failure(let errnoValue):
            logger.error("Failed to open socket: \(errnoValue)")
            return false
        }
        defer {
            close(sock)
        }
        
        var iflpr = if_linkparamsreq()
        var name: (CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        // convert interface name into a tuple of CChars and set in if_linkparamsreq
        let capacity = Mirror(reflecting: name).children.count
        withUnsafeMutablePointer(to: &name) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: capacity) {
                let bound = $0 as UnsafeMutablePointer<CChar>
                bsdName.utf8.enumerated().forEach { (bound + $0.offset).pointee = CChar($0.element) }
            }
        }
        iflpr.iflpr_name = name
        
        var status = ioctl(sock, kSIOCGIFLINKPARAMS, &iflpr)
        guard status == 0 else {
            logger.error("Failed to get link params for interface \(bsdName): \(Errno.current)")
            return false
        }
        logger.debug("Current bandwidth limit: \(iflpr.iflpr_input_netem.ifnetem_bandwidth_bps)")
        logger.debug("Current packet scheduler model: \(iflpr.iflpr_input_netem.ifnetem_model.rawValue)")
        
        // If we are unsetting the bandwidth limit, zero out the input netem params
        if bandwidthLimit == 0 {
            iflpr.iflpr_input_netem = if_netem_params()
        } else {
            iflpr.iflpr_input_netem.ifnetem_model = IF_NETEM_MODEL_NLC
            iflpr.iflpr_input_netem.ifnetem_bandwidth_bps = bandwidthLimit
        }
        
        status = ioctl(sock, kSIOCSIFLINKPARAMS, &iflpr)
        guard status == 0 else {
            logger.error("Failed to set link params for interface \(bsdName): \(Errno.current)")
            return false
        }
        return true
    }

    /// Writes the uplink MTU preference for `mantaremoteagentd`.
    static func setUplinkMTU(_ mtu: Int) -> Int? {

        let domain = CFPreferences.Domain(
            applicationId: kMantaRemoteAgentdBundleId as CFString,
            userName: kCFPreferencesAnyUser,
            hostName: kCFPreferencesAnyHost
        )

        do {
            try CFPreferences.setVerified(
                value: mtu as CFNumber,
                for: kUplinkMTUHintKey,
                in: domain
            )
        } catch {
            Self.logger.error("Failed to set the uplink MTU: \(error, privacy: .public)")
            return nil
        }

        return mtu
    }
}

extension Network {
    struct UplinkInterfaceError: Swift.Error, CustomStringConvertible {
        var description: String

        init(_ description: String) {
            self.description = description
        }
    }
}

extension Network {
    internal enum Error: Swift.Error {
        case incomplete
        case connectionError(Swift.Error)
        case noResponse
        case badResponse(Int)
        case noData
        case dataTransformFailed(Swift.Error)
    }
}

extension Network.Error: LocalizedError {
    internal var errorDescription: String? {
        switch self {
        case .incomplete:
            return "Failed to complete network request"
        case let .connectionError(error):
            return "Connection failed: \(error.localizedDescription)"
        case .noResponse:
            return "Received no response from server"
        case let .badResponse(code):
            return "Received bad response \(code) from server"
        case .noData:
            return "Received no data from server"
        case let .dataTransformFailed(error):
            return "Failed to handle received data: \(error.localizedDescription)"
        }
    }
}

extension Network.Error {
    internal static let retryHTTPCodes = [
        429, // Too Many Requests
        500, // Internal Server Error
        502, // Bad Gateway
        503, // Service Unavailable
        504, // Gateway Timeout
        509, // Bandwidth Limit Exceeded
    ]

    internal var shouldRetry: Bool {
        switch self {
        case .incomplete, .connectionError, .noResponse, .noData:
            return true
        case .badResponse(let code):
            return Self.retryHTTPCodes.contains(code)
        default:
            return false
        }
    }
}

extension URLRequest {
    mutating func addHeaders(additionalHTTPHeaders: [String: String]) {
        for (key, value) in additionalHTTPHeaders {
            self.addValue(value, forHTTPHeaderField: key)
        }
    }
}

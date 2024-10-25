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

import CryptoKit
import Foundation
import os

// dateFormat is normalized form used in tool when displaying dates [YYYY-MM-DDTHH:mm:SS] in local TZ
private let dateFormat = Date.ISO8601FormatStyle(timeZone: TimeZone.current)
    .year().month().day().dateSeparator(.dash)
    .time(includingFractionalSeconds: false).timeSeparator(.colon)

// dateAsString returns a normalized string form of date (relative to epoch)
func dateAsString(_ epoch: UInt64 = UInt64(Date().timeIntervalSince1970)) -> String {
    var date: Date

    if epoch > UInt32.max { // ms
        date = Date(timeIntervalSince1970: Double(epoch) / 1000)
    } else {
        date = Date(timeIntervalSince1970: Double(epoch)) // sec
    }

    return dateAsString(date)
}

// dateAsString returns a formated string form of date
func dateAsString(_ date: Date) -> String {
    return date.formatted(dateFormat)
}

// asJSONString returns (Codable) data as json string, or empty ("{}") if unable to;
//  if pretty == true, result is "pretty printed" with newlines and sorted keys
func asJSONString(_ data: (any Codable)?, pretty: Bool = false) -> String {
    if let data {
        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = pretty ?
            [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes] :
            [.withoutEscapingSlashes]

        if let jsonData = try? jsonEncoder.encode(data) {
            return String(decoding: jsonData, as: UTF8.self)
        }
    }

    return "{}"
}

// ExecCommand provides means to synchronously execution an external command with different means of capturing output
struct ExecCommand {
    private let process: Process

    init(_ command: [String], envvars: [String: String]? = nil) {
        var command = command
        self.process = Process()
        process.qualityOfService = .userInteractive
        process.executableURL = URL(fileURLWithPath: command.removeFirst())
        process.arguments = command
        if let envvars {
            process.environment = envvars
        }
    }

    enum OutputMode {
        case none // no output captured
        case terminal // output set to original stdout/err of caller
        case capture // stdout captured in return arg (stderr suppressed)
        case tee // capture + write to stdout
    }

    @discardableResult
    func run(
        outputMode: OutputMode = .none,
        queue: DispatchQueue? = nil
    ) throws -> (Int32, String, String) {
        let stdoutData = NSMutableData()
        let stderrData = NSMutableData()

        switch outputMode {
        case .none:
            process.standardInput = nil
            process.standardOutput = nil
            process.standardError = nil

        case .capture, .tee:
            process.standardInput = nil
            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            stdoutPipe.fileHandleForReading.readabilityHandler = {
                let data = $0.availableData
                if !data.isEmpty {
                    stdoutData.append(data)
                    if outputMode == .tee {
                        fputs(String(decoding: data, as: UTF8.self), stdout)
                    }
                }
            }

            stderrPipe.fileHandleForReading.readabilityHandler = {
                let data = $0.availableData
                if !data.isEmpty {
                    stderrData.append(data)
                    if outputMode == .tee {
                        fputs(String(decoding: data, as: UTF8.self), stderr)
                    }
                }
            }

        case .terminal:
            // standard stdin/out/err configuration - leave them be
            break
        }

        // signal handling: ensure sub process terminated upon interrupt
        var sigSource: [Int32: DispatchSourceSignal] = [:] // handlers must remain in scope
        if let queue {
            for sigVal in [SIGHUP, SIGINT, SIGQUIT, SIGTERM] {
                signal(sigVal, SIG_IGN)
                sigSource[sigVal] = DispatchSource.makeSignalSource(signal: sigVal, queue: queue)
                sigSource[sigVal]!.setEventHandler { process.terminate() }
                sigSource[sigVal]!.resume()
            }
        }

        try process.run()
        process.waitUntilExit()
        return (process.terminationStatus,
                String(decoding: stdoutData as Data, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                String(decoding: stderrData as Data, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

// computeDigest calculates hash of atPath (file) with hash function (using) returning digest;
//  file is processed in (1 MiB) chunks
func computeDigest(at: URL, using hashFunction: any HashFunction.Type) throws -> any Digest {
    let fhandle = try FileHandle(forReadingFrom: at)
    var hasher = hashFunction.init()

    while autoreleasepool(invoking: {
        guard let nextChunk = try? fhandle.read(upToCount: 1048576) else {
            return false
        }
        hasher.update(data: nextChunk)
        return true
    }) {}

    return hasher.finalize()
}

// dumpURLResponse outputs contents of response from a URLSession call to debug log channel (trace level)
func dumpURLResponse(logger: Logger? = nil, response: URLResponse) {
    guard let logger else {
        return
    }

    func _dlog(_ msg: String) {
        logger.debug("\(msg, privacy: .public)")
    }

    _dlog("URL Response:")
    _dlog("  URL: \(response.url?.absoluteString ?? "unset")")
    _dlog("  mimeType: \(response.mimeType ?? "unset")")
    _dlog("  expectedContentLength: \(response.expectedContentLength)")
    if let suggestedFilename = response.suggestedFilename {
        _dlog("  suggestedFilename: \(suggestedFilename)")
    }
    if let textEncodingName = response.textEncodingName {
        _dlog("  textEncodingName: \(textEncodingName)")
    }

    if let httpResponse = response as? HTTPURLResponse {
        _dlog("  Status: \(httpResponse.statusCode)")
        _dlog("  Headers:")
        for (h, v) in httpResponse.allHeaderFields {
            _dlog("    \(h as? String): \(v as? String)")
        }
    }
}

// getURL performs a simple GET request against url and returns payload; throws an error
//  if request fails, doesn't obtain a 2xx response, or doesn't match provided mimeType
func getURL(
    logger: Logger? = nil,
    url: URL,
    tlsInsecure: Bool = false,
    headers: [String: String]? = nil,
    timeout: TimeInterval = 15,
    mimeType: String? = nil
) async throws -> (Data, URLResponse) {
    var request = URLRequest(url: url, timeoutInterval: timeout)
    addHeaders(&request, headers: headers)

    return try await requestURL(
        logger: logger,
        request: request,
        tlsInsecure: tlsInsecure,
        contentType: mimeType
    )
}

// postPBURL performs a POST request against url with requestBody containing a serialized protobuf
//  and returns (serialized protobuf) payload; throws an error if request fails, doesn't obtain a
//  2xx response, or response content type != "application/protobuf"
func postPBURL(
    logger: Logger? = nil,
    url: URL,
    tlsInsecure: Bool = false,
    requestBody: Data,
    headers: [String: String]? = nil,
    timeout: TimeInterval = 15
) async throws -> (Data, URLResponse) {
    let pbContentType = "application/protobuf"

    var request = URLRequest(url: url, timeoutInterval: timeout)
    request.httpMethod = "POST"
    request.setValue(pbContentType, forHTTPHeaderField: "Content-Type")
    request.httpBody = requestBody
    addHeaders(&request, headers: headers)

    return try await requestURL(
        logger: logger,
        request: request,
        tlsInsecure: tlsInsecure,
        contentType: pbContentType
    )
}

// addHeaders sets headers in request
func addHeaders(_ request: inout URLRequest, headers: [String: String]? = nil) {
    if let headers {
        for (h, v) in headers {
            request.addValue(v, forHTTPHeaderField: h)
        }
    }
}

// requestURL issues populated request to endpoint and returns payload and response if status code 2xx is
//  received and (if contantType is set) confirms mimeType in payload matches expected.
//  TLS verification is suppressed if tlsInsecure is true.
func requestURL(
    logger: Logger? = nil,
    request: URLRequest,
    tlsInsecure: Bool = false,
    contentType: String? = nil
) async throws -> (Data, URLResponse) {
    let session = tlsInsecure ?
        URLSession(configuration: .default,
                   delegate: InsecureTLSDelegate(),
                   delegateQueue: nil) :
        URLSession.shared

    let (respData, response) = try await session.data(for: request)
    dumpURLResponse(logger: logger, response: response)

    guard let httpResponse = response as? HTTPURLResponse else {
        throw URLError(.badServerResponse, userInfo: ["reason": "failed to get response"])
    }

    guard (200 ... 299).contains(httpResponse.statusCode) else {
        throw URLError(.badServerResponse, userInfo: ["reason": "request failed (error: \(httpResponse.statusCode))"])
    }

    if let contentType {
        guard httpResponse.mimeType == contentType else {
            throw URLError(.cannotParseResponse,
                           userInfo: ["reason": "request returned contentType=\(httpResponse.mimeType ?? "unset")"])
        }
    }

    return (respData, response)
}

// InsecureTLSDelegate is a URLSessionDelegate to bypass certificate errors on TLS connections
private class InsecureTLSDelegate: NSObject, URLSessionDelegate {
    public func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust {
            completionHandler(.useCredential,
                              URLCredential(trust: challenge.protectionSpace.serverTrust!))
        } else {
            completionHandler(.performDefaultHandling, nil)
        }
    }
}

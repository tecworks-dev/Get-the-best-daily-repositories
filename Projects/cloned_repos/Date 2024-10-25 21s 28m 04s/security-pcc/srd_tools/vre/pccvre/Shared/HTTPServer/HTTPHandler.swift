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

//===----------------------------------------------------------------------===//
//
// This source file is part of the SwiftNIO open source project
//
// Copyright (c) 2017-2021 Apple Inc. and the SwiftNIO project authors
// Licensed under Apache License v2.0
//
// See LICENSE.txt for license information
// See CONTRIBUTORS.txt for the list of SwiftNIO project authors
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

// Stripped down/modified from SwiftNIO 2.72.0 sample server (Sources/NIOHTTP1Server/main.swift)
// -- only serves files from htdocs
// Request: OSS-20567

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import os

fileprivate extension String {
    func containsDotDot() -> Bool {
        for idx in self.indices {
            if self[idx] == "." &&
                idx < self.index(before: self.endIndex) &&
                self[self.index(after: idx)] == "."
            {
                return true
            }
        }

        return false
    }
}

final class HTTPHandler: ChannelInboundHandler {
    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    private enum State: String {
        case idle
        case waitingForRequestBody
        case sendingResponse

        mutating func requestReceived() {
            guard self == .idle else {
                let cur = self
                HTTPServer.logger.error("invalid state for request received (\(cur.rawValue))")
                self = .idle
                return
            }

            self = .waitingForRequestBody
        }

        mutating func requestComplete() {
            guard self == .waitingForRequestBody else {
                let cur = self
                HTTPServer.logger.error("invalid state for request complete (\(cur.rawValue))")
                self = .idle
                return
            }

            self = .sendingResponse
        }

        mutating func responseComplete() {
            guard self == .sendingResponse else {
                let cur = self
                HTTPServer.logger.error("invalid state for response complete (\(cur.rawValue))")
                self = .idle
                return
            }

            self = .idle
        }
    }

    private var keepAlive = false
    private var state = State.idle
    private let htdocsPath: String

    private var infoSavedRequestHead: HTTPRequestHead?
    private var infoSavedBodyBytes: Int = 0

    private var continuousCount: Int = 0

    private var handler: ((ChannelHandlerContext, HTTPServerRequestPart) -> Void)?
    private var handlerFuture: EventLoopFuture<Void>?
    private let fileIO: NonBlockingFileIO

    public init(fileIO: NonBlockingFileIO, htdocsPath: String) {
        self.htdocsPath = htdocsPath
        self.fileIO = fileIO
    }

    private func httpResponseHead(request: HTTPRequestHead,
                                  status: HTTPResponseStatus,
                                  headers: HTTPHeaders = HTTPHeaders()) -> HTTPResponseHead
    {
        var head = HTTPResponseHead(version: request.version, status: status, headers: headers)
        let connectionHeaders: [String] = head.headers[canonicalForm: "connection"].map { $0.lowercased() }

        if !connectionHeaders.contains("keep-alive") && !connectionHeaders.contains("close") {
            // the user hasn't pre-set either 'keep-alive' or 'close', so we might need to add headers

            switch (request.isKeepAlive, request.version.major, request.version.minor) {
            case (true, 1, 0):
                // HTTP/1.0 and the request has 'Connection: keep-alive', we should mirror that
                head.headers.add(name: "Connection", value: "keep-alive")
            case (false, 1, let n) where n >= 1:
                // HTTP/1.1 (or treated as such) and the request has 'Connection: close', we should mirror that
                head.headers.add(name: "Connection", value: "close")
            default:
                // we should match the default or are dealing with some HTTP that we don't support, let's leave as is
                ()
            }
        }
        return head
    }

    private func handleFile(context: ChannelHandlerContext,
                            request: HTTPServerRequestPart,
                            path: String)
    {
        func sendErrorResponse(request: HTTPRequestHead, _ error: Error) {
            var body = context.channel.allocator.buffer(capacity: 128)
            let response = { () -> HTTPResponseHead in
                let errmsg: String
                let httpResp: HTTPResponseHead
                switch error {
                case let e as IOError where e.errnoCode == ENOENT:
                    errmsg = "IOError (not found)"
                    body.writeString("\(errmsg)\r\n")
                    httpResp = self.httpResponseHead(request: request, status: .notFound)
                case let e as IOError where e.errnoCode == EACCES:
                    errmsg = "Forbidden"
                    body.writeString("\(errmsg)\r\n")
                    httpResp = self.httpResponseHead(request: request, status: .forbidden)
                case let e as IOError:
                    body.writeStaticString("IOError (other)\r\n")
                    body.writeString(e.description)
                    body.writeStaticString("\r\n")
                    errmsg = "IOError (other): \(e.description)"
                    httpResp = self.httpResponseHead(request: request, status: .notFound)
                default:
                    errmsg = "\(type(of: error)) error"
                    body.writeString("\(errmsg)\r\n")
                    httpResp = self.httpResponseHead(request: request, status: .internalServerError)
                }

                HTTPServer.logger.error("\(request, privacy: .public): \(errmsg, privacy: .public)")
                return httpResp
            }()
            context.write(Self.wrapOutboundOut(.head(response)), promise: nil)
            context.write(Self.wrapOutboundOut(.body(.byteBuffer(body))), promise: nil)
            context.writeAndFlush(Self.wrapOutboundOut(.end(nil)), promise: nil)
            context.channel.close(promise: nil)
        }

        func responseHead(request: HTTPRequestHead, fileRegion region: FileRegion) -> HTTPResponseHead {
            var response = self.httpResponseHead(request: request, status: .ok)
            response.headers.add(name: "Content-Length", value: "\(region.endIndex)")
            response.headers.add(name: "Content-Type", value: "application/octet-stream")
            return response
        }

        switch request {
        case .head(let request):
            HTTPServer.logger.log("request: \(request, privacy: .public)")
            self.keepAlive = request.isKeepAlive
            self.state.requestReceived()
            guard !request.uri.containsDotDot() && request.method == .GET else {
                sendErrorResponse(request: request, IOError(errnoCode: EACCES, reason: "forbidden"))
                return
            }

            let path = self.htdocsPath + "/" + path
            HTTPServer.logger.debug("request: full path: \(path, privacy: .public)")
            let fileHandleAndRegion = self.fileIO.openFile(path: path, eventLoop: context.eventLoop)
            fileHandleAndRegion.whenFailure { error in
                sendErrorResponse(request: request, error)
            }
            fileHandleAndRegion.whenSuccess { file, region in
                var responseStarted = false
                let response = responseHead(request: request, fileRegion: region)
                if region.readableBytes == 0 {
                    HTTPServer.logger.log("\(request.uri, privacy: .public): \(response, privacy: .public)")
                    responseStarted = true
                    context.write(Self.wrapOutboundOut(.head(response)), promise: nil)
                }

                return self.fileIO.readChunked(fileRegion: region,
                                               chunkSize: 32 * 1024,
                                               allocator: context.channel.allocator,
                                               eventLoop: context.eventLoop)
                { buffer in
                    if !responseStarted {
                        responseStarted = true
                        context.write(Self.wrapOutboundOut(.head(response)), promise: nil)
                    }
                    return context.writeAndFlush(Self.wrapOutboundOut(.body(.byteBuffer(buffer))))
                }.flatMap { () -> EventLoopFuture<Void> in
                    let p = context.eventLoop.makePromise(of: Void.self)
                    self.completeResponse(context, trailers: nil, promise: p)
                    return p.futureResult
                }.flatMapError { error in
                    if !responseStarted {
                        sendErrorResponse(request: request, error)
                        self.state.responseComplete()
                        return context.writeAndFlush(Self.wrapOutboundOut(.end(nil)))
                    } else {
                        return context.close()
                    }
                }.whenComplete { (_: Result<Void, Error>) in
                    _ = try? file.close()
                }
            }
        case .end:
            HTTPServer.logger.log("request: END")
            self.state.requestComplete()
        default:
            HTTPServer.logger.error("request: UNKNOWN")
            self.completeResponse(context, trailers: nil, promise: nil)
        }
    }

    private func completeResponse(_ context: ChannelHandlerContext,
                                  trailers: HTTPHeaders?,
                                  promise: EventLoopPromise<Void>?)
    {
        self.state.responseComplete()

        let promise = self.keepAlive ? promise : (promise ?? context.eventLoop.makePromise())
        if !self.keepAlive {
            promise!.futureResult.whenComplete { (_: Result<Void, Error>) in context.close(promise: nil) }
        }
        self.handler = nil

        context.writeAndFlush(Self.wrapOutboundOut(.end(trailers)), promise: promise)
        HTTPServer.logger.log("completeResponse")
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let reqPart = Self.unwrapInboundIn(data)
        if let handler = self.handler {
            handler(context, reqPart)
            return
        }

        switch reqPart {
        case .head(let request):
            // set handler (in case we add new functionality later beyond file requests)
            self.handler = {
                self.handleFile(
                    context: $0,
                    request: $1,
                    path: request.uri.removingPercentEncoding ?? request.uri)
            }
            self.handler!(context, reqPart)
        case .body:
            break
        case .end:
            self.completeResponse(context, trailers: nil, promise: nil)
        }
    }

    func channelReadComplete(context: ChannelHandlerContext) {
        context.flush()
    }

    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        switch event {
        case let evt as ChannelEvent where evt == ChannelEvent.inputClosed:
            // The remote peer half-closed the channel. At this time, any
            // outstanding response will now get the channel closed, and
            // if we are idle or waiting for a request body to finish we
            // will close the channel immediately.
            switch self.state {
            case .idle, .waitingForRequestBody:
                context.close(promise: nil)
            case .sendingResponse:
                self.keepAlive = false
            }
        default:
            context.fireUserInboundEventTriggered(event)
        }
    }
}

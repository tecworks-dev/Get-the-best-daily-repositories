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

import Foundation
import Network
import NIOCore
import NIOHTTP1
import NIOPosix
import os

// Parameters for HTTP service and launcher

struct HTTPServer {
    let docDir: String // HTTP docroot (typ VRE instance dir)
    let uriMethod: String = "http"
    let bindAddr: IPAddress
    var bindPort: UInt16?

    static let logger = os.Logger(subsystem: applicationName, category: "HTTPServer")

    private var vmeBridge: VMEnetBridge? // set when binding to vmenet bridge
    private var channel: Channel? // channel used by http service
    private var task: Task<Void, Error>? // thread containing http service

    init(
        docDir: String,
        bindAddr: String?, // determine automatically (vmenet bridge vif) if nil
        bindPort: UInt16? // os selected if nil
    ) async throws {
        self.docDir = docDir

        if let bindAddr {
            guard let ipaddr: IPAddress = IPv4Address(bindAddr) ?? IPv6Address(bindAddr) else {
                throw HTTPServerError("invalid bindAddr")
            }

            self.bindAddr = ipaddr
        } else {
            // otherwise, bring up a vmenet (NET) interface to plumb the host/vm "bridge" vif and glean its IP
            var vmeBridge = VMEnetBridge()
            do {
                try await vmeBridge.bringup()
            } catch {
                throw HTTPServerError("could not bringup vmenet interface: \(error)")
            }

            guard let bridgeIP = vmeBridge.ip4Addr else {
                try? vmeBridge.shutdown()
                throw HTTPServerError("could not determine IP address of vmenet bridge")
            }

            self.vmeBridge = vmeBridge
            self.bindAddr = bridgeIP
        }

        self.bindPort = bindPort
    }

    mutating func start() throws {
        let docDir = self.docDir
        let fileIO = NonBlockingFileIO(threadPool: .singleton)
        func childChannelInitializer(channel: Channel) -> EventLoopFuture<Void> {
            return channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
                channel.pipeline.addHandler(HTTPHandler(fileIO: fileIO, htdocsPath: docDir))
            }
        }

        let socketBootstrap = ServerBootstrap(group: MultiThreadedEventLoopGroup.singleton)
            // Specify backlog and enable SO_REUSEADDR for the server itself
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            // Set the handlers that are applied to the accepted Channels
            .childChannelInitializer(childChannelInitializer(channel:))
            // Enable SO_REUSEADDR for the accepted Channels
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.allowRemoteHalfClosure, value: true)

        let bindAddr = String(describing: self.bindAddr)
        let bindPort = Int(self.bindPort ?? 0)

        let channel: Channel
        do {
            channel = try socketBootstrap.bind(host: bindAddr, port: bindPort).wait()
        } catch {
            throw HTTPServerError("could not bind to \(bindAddr):\(bindPort): \(error)")
        }

        guard let bindPort = channel.localAddress?.port else {
            throw HTTPServerError("could not determine local address")
        }

        let endpoint = "\(bindAddr):\(bindPort)"
        HTTPServer.logger.log("server instance \(endpoint, privacy: .public)")

        self.bindPort = UInt16(bindPort)
        self.channel = channel

        self.task = Task {
            

            // This will never unblock as we don't close the ServerChannel
            try await channel.closeFuture.get()
        }
    }

    mutating func shutdown() throws {
        if let task = self.task {
            task.cancel()
        }

        if var vmeBridge {
            try? vmeBridge.shutdown()
        }

        self.task = nil
        self.vmeBridge = nil
    }

    // baseURL returns root (base) endpoint for this HTTPServer instance
    func baseURL() -> URL? {
        let bindAddr = String(describing: self.bindAddr)
        var baseURL = "http://\(bindAddr)"
        if let bindPort {
            baseURL += ":\(bindPort)"
        }

        return URL(string: baseURL)
    }

    // makeURL returns URL of path relative to baseURL; if path already represents a URL,
    //  it is returned instead
    func makeURL(path: String) -> URL? {
        if let baseURL = baseURL() {
            return URL(string: path, relativeTo: baseURL)
        }

        return nil
    }
}

// HTTPServerError provides general error encapsulation for errors encountered in HTTPServer layer
struct HTTPServerError: Error, CustomStringConvertible {
    var message: String
    var description: String { self.message }

    init(_ message: String) {
        HTTPServer.logger.error("\(message, privacy: .public)")
        self.message = message
    }
}

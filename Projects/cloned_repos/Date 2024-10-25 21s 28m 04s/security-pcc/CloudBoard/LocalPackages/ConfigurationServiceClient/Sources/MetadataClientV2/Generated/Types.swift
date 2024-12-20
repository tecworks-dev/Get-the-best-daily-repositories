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

// Generated by swift-openapi-generator, do not modify.
@_spi(Generated) import OpenAPIRuntime
#if os(Linux)
@preconcurrency import struct Foundation.URL
@preconcurrency import struct Foundation.Data
@preconcurrency import struct Foundation.Date
#else
import struct Foundation.URL
import struct Foundation.Data
import struct Foundation.Date
#endif
/// A type that performs HTTP operations defined by the OpenAPI document.
package protocol APIProtocol: Sendable {
    /// Return the current revision control that the consumer should be using.
    ///
    /// Returns 200 with the current control for the given project, environment, and release. If the optional instanceId and instanceRevision query parameters are provided, returns 200 with the current control if instanceId should download a new config or 304 if instanceId should continue using config with instanceRevision.
    /// the current revision is different.
    ///
    /// - Remark: HTTP `GET /controls/project/{project}/environment/{environment}/release/{release}/current`.
    /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)`.
    func getCurrentRevisionControl(_ input: Operations.getCurrentRevisionControl.Input) async throws -> Operations.getCurrentRevisionControl.Output
}

/// Convenience overloads for operation inputs.
extension APIProtocol {
    /// Return the current revision control that the consumer should be using.
    ///
    /// Returns 200 with the current control for the given project, environment, and release. If the optional instanceId and instanceRevision query parameters are provided, returns 200 with the current control if instanceId should download a new config or 304 if instanceId should continue using config with instanceRevision.
    /// the current revision is different.
    ///
    /// - Remark: HTTP `GET /controls/project/{project}/environment/{environment}/release/{release}/current`.
    /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)`.
    package func getCurrentRevisionControl(
        path: Operations.getCurrentRevisionControl.Input.Path,
        query: Operations.getCurrentRevisionControl.Input.Query = .init(),
        headers: Operations.getCurrentRevisionControl.Input.Headers = .init()
    ) async throws -> Operations.getCurrentRevisionControl.Output {
        try await getCurrentRevisionControl(Operations.getCurrentRevisionControl.Input(
            path: path,
            query: query,
            headers: headers
        ))
    }
}

/// Server URLs defined in the OpenAPI document.
package enum Servers {}

/// Types generated from the components section of the OpenAPI document.
package enum Components {
    /// Types generated from the `#/components/schemas` section of the OpenAPI document.
    package enum Schemas {
        /// - Remark: Generated from `#/components/schemas/Control`.
        package struct Control: Codable, Hashable, Sendable {
            /// The name of the project.
            ///
            /// - Remark: Generated from `#/components/schemas/Control/project`.
            package var project: Swift.String
            /// The identifier of the release.
            ///
            /// - Remark: Generated from `#/components/schemas/Control/release`.
            package var release: Swift.String
            /// The identifier of the revision.
            ///
            /// - Remark: Generated from `#/components/schemas/Control/revision`.
            package var revision: Swift.String
            /// Creates a new `Control`.
            ///
            /// - Parameters:
            ///   - project: The name of the project.
            ///   - release: The identifier of the release.
            ///   - revision: The identifier of the revision.
            package init(
                project: Swift.String,
                release: Swift.String,
                revision: Swift.String
            ) {
                self.project = project
                self.release = release
                self.revision = revision
            }
            package enum CodingKeys: String, CodingKey {
                case project
                case release
                case revision
            }
        }
    }
    /// Types generated from the `#/components/parameters` section of the OpenAPI document.
    package enum Parameters {}
    /// Types generated from the `#/components/requestBodies` section of the OpenAPI document.
    package enum RequestBodies {}
    /// Types generated from the `#/components/responses` section of the OpenAPI document.
    package enum Responses {}
    /// Types generated from the `#/components/headers` section of the OpenAPI document.
    package enum Headers {}
}

/// API operations, with input and output types, generated from `#/paths` in the OpenAPI document.
package enum Operations {
    /// Return the current revision control that the consumer should be using.
    ///
    /// Returns 200 with the current control for the given project, environment, and release. If the optional instanceId and instanceRevision query parameters are provided, returns 200 with the current control if instanceId should download a new config or 304 if instanceId should continue using config with instanceRevision.
    /// the current revision is different.
    ///
    /// - Remark: HTTP `GET /controls/project/{project}/environment/{environment}/release/{release}/current`.
    /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)`.
    package enum getCurrentRevisionControl {
        package static let id: Swift.String = "getCurrentRevisionControl"
        package struct Input: Sendable, Hashable {
            /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/path`.
            package struct Path: Sendable, Hashable {
                /// The name of the project.
                ///
                /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/path/project`.
                package var project: Swift.String
                /// The name of the environment.
                ///
                /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/path/environment`.
                package var environment: Swift.String
                /// The identifier of the release.
                ///
                /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/path/release`.
                package var release: Swift.String
                /// Creates a new `Path`.
                ///
                /// - Parameters:
                ///   - project: The name of the project.
                ///   - environment: The name of the environment.
                ///   - release: The identifier of the release.
                package init(
                    project: Swift.String,
                    environment: Swift.String,
                    release: Swift.String
                ) {
                    self.project = project
                    self.environment = environment
                    self.release = release
                }
            }
            package var path: Operations.getCurrentRevisionControl.Input.Path
            /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/query`.
            package struct Query: Sendable, Hashable {
                /// The unique name of the instance.
                ///
                /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/query/instanceId`.
                package var instanceId: Swift.String?
                /// The revision of the configuration the instance is running.
                ///
                /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/query/instanceRevision`.
                package var instanceRevision: Swift.String?
                /// Creates a new `Query`.
                ///
                /// - Parameters:
                ///   - instanceId: The unique name of the instance.
                ///   - instanceRevision: The revision of the configuration the instance is running.
                package init(
                    instanceId: Swift.String? = nil,
                    instanceRevision: Swift.String? = nil
                ) {
                    self.instanceId = instanceId
                    self.instanceRevision = instanceRevision
                }
            }
            package var query: Operations.getCurrentRevisionControl.Input.Query
            /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/header`.
            package struct Headers: Sendable, Hashable {
                package var accept: [OpenAPIRuntime.AcceptHeaderContentType<Operations.getCurrentRevisionControl.AcceptableContentType>]
                /// Creates a new `Headers`.
                ///
                /// - Parameters:
                ///   - accept:
                package init(accept: [OpenAPIRuntime.AcceptHeaderContentType<Operations.getCurrentRevisionControl.AcceptableContentType>] = .defaultValues()) {
                    self.accept = accept
                }
            }
            package var headers: Operations.getCurrentRevisionControl.Input.Headers
            /// Creates a new `Input`.
            ///
            /// - Parameters:
            ///   - path:
            ///   - query:
            ///   - headers:
            package init(
                path: Operations.getCurrentRevisionControl.Input.Path,
                query: Operations.getCurrentRevisionControl.Input.Query = .init(),
                headers: Operations.getCurrentRevisionControl.Input.Headers = .init()
            ) {
                self.path = path
                self.query = query
                self.headers = headers
            }
        }
        @frozen package enum Output: Sendable, Hashable {
            package struct Ok: Sendable, Hashable {
                /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/responses/200/content`.
                @frozen package enum Body: Sendable, Hashable {
                    /// - Remark: Generated from `#/paths/controls/project/{project}/environment/{environment}/release/{release}/current/GET/responses/200/content/application\/json`.
                    case json(Components.Schemas.Control)
                    /// The associated value of the enum case if `self` is `.json`.
                    ///
                    /// - Throws: An error if `self` is not `.json`.
                    /// - SeeAlso: `.json`.
                    package var json: Components.Schemas.Control {
                        get throws {
                            switch self {
                            case let .json(body):
                                return body
                            }
                        }
                    }
                }
                /// Received HTTP response body
                package var body: Operations.getCurrentRevisionControl.Output.Ok.Body
                /// Creates a new `Ok`.
                ///
                /// - Parameters:
                ///   - body: Received HTTP response body
                package init(body: Operations.getCurrentRevisionControl.Output.Ok.Body) {
                    self.body = body
                }
            }
            /// The current control.
            ///
            /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)/responses/200`.
            ///
            /// HTTP response code: `200 ok`.
            case ok(Operations.getCurrentRevisionControl.Output.Ok)
            /// The associated value of the enum case if `self` is `.ok`.
            ///
            /// - Throws: An error if `self` is not `.ok`.
            /// - SeeAlso: `.ok`.
            package var ok: Operations.getCurrentRevisionControl.Output.Ok {
                get throws {
                    switch self {
                    case let .ok(response):
                        return response
                    default:
                        try throwUnexpectedResponseStatus(
                            expectedStatus: "ok",
                            response: self
                        )
                    }
                }
            }
            package struct NotModified: Sendable, Hashable {
                /// Creates a new `NotModified`.
                package init() {}
            }
            /// There is no new config available for the provided instanceId running the config with instanceRevision, no action needed.
            ///
            /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)/responses/304`.
            ///
            /// HTTP response code: `304 notModified`.
            case notModified(Operations.getCurrentRevisionControl.Output.NotModified)
            /// The associated value of the enum case if `self` is `.notModified`.
            ///
            /// - Throws: An error if `self` is not `.notModified`.
            /// - SeeAlso: `.notModified`.
            package var notModified: Operations.getCurrentRevisionControl.Output.NotModified {
                get throws {
                    switch self {
                    case let .notModified(response):
                        return response
                    default:
                        try throwUnexpectedResponseStatus(
                            expectedStatus: "notModified",
                            response: self
                        )
                    }
                }
            }
            package struct BadRequest: Sendable, Hashable {
                /// Creates a new `BadRequest`.
                package init() {}
            }
            /// Bad request caused by invalid inputs.
            ///
            /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)/responses/400`.
            ///
            /// HTTP response code: `400 badRequest`.
            case badRequest(Operations.getCurrentRevisionControl.Output.BadRequest)
            /// The associated value of the enum case if `self` is `.badRequest`.
            ///
            /// - Throws: An error if `self` is not `.badRequest`.
            /// - SeeAlso: `.badRequest`.
            package var badRequest: Operations.getCurrentRevisionControl.Output.BadRequest {
                get throws {
                    switch self {
                    case let .badRequest(response):
                        return response
                    default:
                        try throwUnexpectedResponseStatus(
                            expectedStatus: "badRequest",
                            response: self
                        )
                    }
                }
            }
            package struct Forbidden: Sendable, Hashable {
                /// Creates a new `Forbidden`.
                package init() {}
            }
            /// Forbidden.
            ///
            /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)/responses/403`.
            ///
            /// HTTP response code: `403 forbidden`.
            case forbidden(Operations.getCurrentRevisionControl.Output.Forbidden)
            /// The associated value of the enum case if `self` is `.forbidden`.
            ///
            /// - Throws: An error if `self` is not `.forbidden`.
            /// - SeeAlso: `.forbidden`.
            package var forbidden: Operations.getCurrentRevisionControl.Output.Forbidden {
                get throws {
                    switch self {
                    case let .forbidden(response):
                        return response
                    default:
                        try throwUnexpectedResponseStatus(
                            expectedStatus: "forbidden",
                            response: self
                        )
                    }
                }
            }
            package struct NotFound: Sendable, Hashable {
                /// Creates a new `NotFound`.
                package init() {}
            }
            /// Resource was not found.
            ///
            /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)/responses/404`.
            ///
            /// HTTP response code: `404 notFound`.
            case notFound(Operations.getCurrentRevisionControl.Output.NotFound)
            /// The associated value of the enum case if `self` is `.notFound`.
            ///
            /// - Throws: An error if `self` is not `.notFound`.
            /// - SeeAlso: `.notFound`.
            package var notFound: Operations.getCurrentRevisionControl.Output.NotFound {
                get throws {
                    switch self {
                    case let .notFound(response):
                        return response
                    default:
                        try throwUnexpectedResponseStatus(
                            expectedStatus: "notFound",
                            response: self
                        )
                    }
                }
            }
            package struct TooManyRequests: Sendable, Hashable {
                /// Creates a new `TooManyRequests`.
                package init() {}
            }
            /// Request was rate-limited.
            ///
            /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)/responses/429`.
            ///
            /// HTTP response code: `429 tooManyRequests`.
            case tooManyRequests(Operations.getCurrentRevisionControl.Output.TooManyRequests)
            /// The associated value of the enum case if `self` is `.tooManyRequests`.
            ///
            /// - Throws: An error if `self` is not `.tooManyRequests`.
            /// - SeeAlso: `.tooManyRequests`.
            package var tooManyRequests: Operations.getCurrentRevisionControl.Output.TooManyRequests {
                get throws {
                    switch self {
                    case let .tooManyRequests(response):
                        return response
                    default:
                        try throwUnexpectedResponseStatus(
                            expectedStatus: "tooManyRequests",
                            response: self
                        )
                    }
                }
            }
            package struct InternalServerError: Sendable, Hashable {
                /// Creates a new `InternalServerError`.
                package init() {}
            }
            /// Internal server error.
            ///
            /// - Remark: Generated from `#/paths//controls/project/{project}/environment/{environment}/release/{release}/current/get(getCurrentRevisionControl)/responses/500`.
            ///
            /// HTTP response code: `500 internalServerError`.
            case internalServerError(Operations.getCurrentRevisionControl.Output.InternalServerError)
            /// The associated value of the enum case if `self` is `.internalServerError`.
            ///
            /// - Throws: An error if `self` is not `.internalServerError`.
            /// - SeeAlso: `.internalServerError`.
            package var internalServerError: Operations.getCurrentRevisionControl.Output.InternalServerError {
                get throws {
                    switch self {
                    case let .internalServerError(response):
                        return response
                    default:
                        try throwUnexpectedResponseStatus(
                            expectedStatus: "internalServerError",
                            response: self
                        )
                    }
                }
            }
            /// Undocumented response.
            ///
            /// A response with a code that is not documented in the OpenAPI document.
            case undocumented(statusCode: Swift.Int, OpenAPIRuntime.UndocumentedPayload)
        }
        @frozen package enum AcceptableContentType: AcceptableProtocol {
            case json
            case other(Swift.String)
            package init?(rawValue: Swift.String) {
                switch rawValue.lowercased() {
                case "application/json":
                    self = .json
                default:
                    self = .other(rawValue)
                }
            }
            package var rawValue: Swift.String {
                switch self {
                case let .other(string):
                    return string
                case .json:
                    return "application/json"
                }
            }
            package static var allCases: [Self] {
                [
                    .json
                ]
            }
        }
    }
}

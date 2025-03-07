//
//  Copyright wildberry Inc. All Rights Reserved.
//
//  Licensed under the MIT License (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      https://opensource.org/licenses/MIT
//
//  PostAdServicesTokenOperation.swift
//
//  Created by Madeline Beyl on 4/20/22.

import Foundation

class PostAdServicesTokenOperation: NetworkOperation {

    private let configuration: UserSpecificConfiguration
    private let token: String
    private let responseHandler: CustomerAPI.SimpleResponseHandler?

    init(configuration: UserSpecificConfiguration,
         token: String,
         responseHandler: CustomerAPI.SimpleResponseHandler?) {
        self.token = token
        self.configuration = configuration
        self.responseHandler = responseHandler

        super.init(configuration: configuration)
    }

    override func begin(completion: @escaping () -> Void) {
        self.post(completion: completion)
    }

    private func post(completion: @escaping () -> Void) {
        let appUserID = self.configuration.appUserID

        guard appUserID.isNotEmpty else {
            self.responseHandler?(.missingAppUserID())
            completion()
            return
        }

        let request = HTTPRequest(method: .post(Body(aadAttributionToken: self.token)),
                                  path: .postAdServicesToken(appUserID: appUserID))

        self.httpClient.perform(request) { (response: VerifiedHTTPResponse<HTTPEmptyResponseBody>.Result) in
            defer {
                completion()
            }

            self.responseHandler?(response.error.map(BackendError.networkError))
        }
    }

}

// Restating inherited @unchecked Sendable from Foundation's Operation
extension PostAdServicesTokenOperation: @unchecked Sendable {}

private extension PostAdServicesTokenOperation {

    struct Body: HTTPRequestBody {

        let aadAttributionToken: String

        init(aadAttributionToken: String) {
            self.aadAttributionToken = aadAttributionToken
        }

    }

}

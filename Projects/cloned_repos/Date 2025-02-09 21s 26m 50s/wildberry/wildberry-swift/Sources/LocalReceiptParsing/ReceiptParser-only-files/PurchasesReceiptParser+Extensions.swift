//
//  Copyright wildberry Inc. All Rights Reserved.
//
//  Licensed under the MIT License (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      https://opensource.org/licenses/MIT
//
//  PurchasesReceiptParser+Extensions.swift
//
//  Created by Nacho Soto on 12/14/22.

import Foundation

public extension PurchasesReceiptParser {

    /// A default ``PurchasesReceiptParser`` configured for use.
    static let `default`: PurchasesReceiptParser = .init(logger: ReceiptParserLogger())

    /// Fetches and parses the local receipt
    /// - Returns: ``AppleReceipt`` of the parsed local receipt.
    /// - Throws: ``PurchasesReceiptParser/Error`` if fetching or parsing failed.
    func fetchAndParseLocalReceipt() throws -> AppleReceipt {
        return try LocalReceiptFetcher().fetchAndParseLocalReceipt()
    }

}

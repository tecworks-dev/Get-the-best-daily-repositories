//
//  Memoji.swift
//
//  Created by James Sedlacek on 3/7/25.
//

import SwiftUI

/// A structure representing a Memoji avatar image used throughout the app.
///
/// Memojis are used to represent users in reviews and other UI components.
/// This struct provides access to a collection of predefined Memoji images.
struct Memoji {
    /// The actual image containing the Memoji graphic.
    let image: Image

    /// Provides indexed access to a collection of predefined Memoji images.
    ///
    /// - Parameter index: The zero-based index of the desired Memoji.
    /// - Returns: A Memoji instance containing the image at the requested index.
    ///            If the index is out of bounds, it safely clamps to the valid range.
    static subscript(index: Int) -> Memoji {
        let options = [
            Image(.person1),
            Image(.person2),
            Image(.person3),
            Image(.person4),
            Image(.person5),
            Image(.person6),
            Image(.person7),
            Image(.person8)
        ]
        let safeIndex = max(0, min(index, options.count - 1))
        return Memoji(image: options[safeIndex])
    }
}

//
//  Photo.swift
//  SplashScreenKit
//  Copyright © 2024 Apple Inc. (https://developer.apple.com/documentation/swiftui/creating-visual-effects-with-swiftui)
//  Modified by Ming on 10/2/2025.
//

import SwiftUI

public struct Photo: Identifiable {
    public var title: String

    public var id: Int = .random(in: 0 ... 100)

    public init(_ title: String) {
        self.title = title
    }
}

public struct ItemPhoto: View {
    public var photo: Photo

    public init(_ photo: Photo) {
        self.photo = photo
    }

    public var body: some View {
        Image(photo.title)
            .resizable()
            .scaledToFill()
            .frame(width: 219, height: 475)
            .clipShape(RoundedRectangle(cornerRadius: 36))
            .shadow(radius: 5)
    }
}

public struct ItemLabel: View {
    public var photo: Photo

    public init(_ photo: Photo) {
        self.photo = photo
    }

    public var body: some View {
        Text(photo.title)
            .font(.title)
    }
}

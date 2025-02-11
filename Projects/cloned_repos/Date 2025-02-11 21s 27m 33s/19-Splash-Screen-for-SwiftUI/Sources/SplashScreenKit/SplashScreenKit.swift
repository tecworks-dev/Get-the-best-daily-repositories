//
//  Created by Ming on 10/2/2025. Inspired by 'Creating visual effects with SwiftUI' & Apple Invites app 2025.
//

import SwiftUI

@available(iOS 18.0, *)
public struct SplashScreen: View {
    @State var prVisible: Bool = false
    @State var ctaVisible: Bool = false
    @State var scrollOffset: CGFloat = 0
    @State var timer: Timer?
    @State var photos: [Photo]
    
    var title: String
    var product: String
    var caption: String
    var ctaText: String
    var ctaAction: () -> Void
    
    public init(images: [Photo], title: String, product: String, caption: String, cta: String, action: @escaping () -> Void) {
        self._photos = State(initialValue: images) // Initialize @State variable
        self.title = title
        self.product = product
        self.caption = caption
        self.ctaText = cta
        self.ctaAction = action
    }
    
    public var body: some View {
        ZStack {
            Image("\(photos[currentIndex].title)")
                .resizable()
                .ignoresSafeArea(.all)
                .blur(radius: 10)
            VStack {
                pagingRotation
                    .offset(y: prVisible ? 0 : -500)
                    .transition(.move(edge: .top))
                    .animation(.easeInOut(duration: 1))
                cta
                Spacer()
            }
            .background(.black.opacity(0.8))
            .background(.ultraThinMaterial)
        }
    }

    public var pagingRotation: some View {
        GeometryReader { geometry in
            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 30) {
                    ForEach(Array(photos).enumerated().map { $0 }, id: \.offset) { index, photo in
                        ItemPhoto(photo)
                            .scrollTransition(axis: .horizontal) { content, phase in
                                content
                                    .rotationEffect(.degrees(phase.value * 2.5))
                                   // Experiental
                                   .scaleEffect(1 - abs(phase.value) * 0.025)
                                   .opacity(1 - abs(phase.value) * 0.8)
                            }
                    }
                }
                .offset(x: scrollOffset)
                .onAppear {
                    startAutoScroll(geometry.size.width)
                }
                .onDisappear {
                    timer?.invalidate()
                    prVisible = false
                    ctaVisible = false
                }
                .onChange(of: currentIndex) { index in
                    withAnimation(nil) {
                        if index >= photos.count - 3 {
                            photos.append(contentsOf: photos)
                        }
                    }
                }
            }.disabled(true)
            .contentMargins(24)
            .frame(height:475)
        }.frame(height:475)
        .padding(.vertical, 25)
    }

    public func startAutoScroll(_ viewWidth: CGFloat) {
        timer = Timer.scheduledTimer(withTimeInterval: 0.02, repeats: true) { _ in
            Task { @MainActor in
                withAnimation {
                    scrollOffset -= 1
                    if scrollOffset <= -viewWidth * CGFloat(photos.count - 1) {
                        scrollOffset = 0
                    }
                }
            }
        }
    }
    public var currentIndex: Int {
        let imageWidth: CGFloat = 219 + 30 // Image width + spacing
        return Int((-scrollOffset + imageWidth / 2) / imageWidth)
    }
    
    public var cta: some View {
            VStack {
                if ctaVisible {
                    Text(title)
                        .font(.system(size: 20, weight: .bold, design: .default))
                        .foregroundStyle(.secondary)
                        .transition(TextTransition())
                    Text(product)
                        .font(.system(size: 50, weight: .bold, design: .default))
                        .customAttribute(EmphasisAttribute())
                        .transition(TextTransition())
                        .padding(.bottom,5)
                    Text(caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .transition(TextTransition())
                    Button(action: ctaAction) {
                        Text(ctaText)
                            .font(.system(size: 15, weight: .bold, design: .default))
                            .padding(.vertical, 10)
                            .padding(.horizontal, 10)
                    }.buttonStyle(.borderedProminent)
                    .buttonBorderShape(.capsule)
                    .tint(.white)
                    .foregroundStyle(.black)
                    .padding(25)
                }
            }.foregroundStyle(.white)
            .padding()
            .onAppear {
                prVisible = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                    ctaVisible = true
                }
            }
            .animation(.easeInOut(duration: 2))
        }
}

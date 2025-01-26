import "./globals.css";
import "../styles/scrollbar.css";

import Footer from "./_components/Footer.js";
import Header from "./_components/Header.js";
import Providers from "./providers.js";
import { Toaster } from "./_components/ui/Toaster.js";

export const metadata = {
  title: "MAIA AI",
  description: "I Scan Twitter to keep you ahead",
};

export default function RootLayout({ children }) {
  return (
    <html>
      <head>
        <link
          rel="icon"
          type="image/png"
          href="/favicon-96x96.png"
          sizes="96x96"
        />
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <link rel="shortcut icon" href="/favicon.ico" />
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href="/apple-touch-icon.png"
        />
        <link rel="manifest" href="/site.webmanifest" />
      </head>
      <body
        suppressHydrationWarning
        data-overlayscrollbars-initialize
        className="dark"
      >
        <Providers>
          <Header />
          {children}
          <Footer />
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}

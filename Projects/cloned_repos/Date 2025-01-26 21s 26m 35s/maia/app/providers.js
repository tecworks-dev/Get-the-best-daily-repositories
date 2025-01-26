"use client";

import { AuthProvider } from "./_providers/AuthProvider.js";
import NUProvider from "./_providers/NUProvider.js";
import { ScrollbarProvider } from "./_providers/ScrollbarProvider.js";

export default function Providers({ children }) {
  return (
    <NUProvider>
      <AuthProvider>
        <ScrollbarProvider>{children}</ScrollbarProvider>
      </AuthProvider>
    </NUProvider>
  );
}

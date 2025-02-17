import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { SessionProvider } from "@/components/providers/SessionProvider";
import { AppNav } from "@/components/navigation/AppNav";
import { DndProvider } from "@/components/dnd/DndProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "FluidCalendar",
  description: "A modern calendar and task management application",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body
        className={cn(
          inter.className,
          "h-full bg-background antialiased",
          "flex flex-col"
        )}
      >
        <SessionProvider>
          <DndProvider>
            <AppNav />
            <main className="flex-1 relative">{children}</main>
          </DndProvider>
        </SessionProvider>
      </body>
    </html>
  );
}

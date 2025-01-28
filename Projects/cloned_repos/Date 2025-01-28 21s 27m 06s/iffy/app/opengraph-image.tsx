import { BuyMeACoffee } from "@/components/logos/buy-me-a-coffee";
import { Gumroad } from "@/components/logos/gumroad";
import { ImageResponse } from "next/og";
import { cache } from "@/lib/cache";

export const runtime = "edge";

export default async function GET() {
  return new ImageResponse(
    (
      <div
        style={{
          background: "white",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "80px",
          fontFamily: "IBM Plex Sans",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "2rem" }}>
          <div
            style={{
              fontSize: 32,
              fontWeight: 500,
              display: "flex",
              gap: "0.5rem",
              fontFamily: "IBM Plex Mono",
              textAlign: "left",
              lineHeight: "26px",
            }}
          >
            <span style={{ letterSpacing: "-0.15em" }}>:/</span>
            <span>iffy</span>
          </div>
          <div style={{ fontSize: 80 }}>Intelligent content moderation at scale</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "20px" }}>
          <div style={{ fontSize: 32, color: "gray", fontWeight: 400 }}>
            Trusted by leading companies to keep the Internet clean and safe:
          </div>
          <div style={{ display: "flex", gap: "10px" }}>
            <BuyMeACoffee />
            <Gumroad />
          </div>
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: "IBM Plex Sans",
          data: await fetch(new URL("@/public/fonts/IBMPlexSans-Medium.ttf", import.meta.url)).then((res) =>
            res.arrayBuffer(),
          ),
          style: "normal",
          weight: 500,
        },
        {
          name: "IBM Plex Mono",
          data: await fetch(new URL("@/public/fonts/IBMPlexMono-Medium.ttf", import.meta.url)).then((res) =>
            res.arrayBuffer(),
          ),
          style: "normal",
          weight: 500,
        },
      ],
    },
  );
}

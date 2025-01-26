"use client";

import { useEffect } from "react";
import { AuroraBackground } from "./_components/ui/AuroraBackground.js";
import Aos from "aos";
import "aos/dist/aos.css";
import Analyzer from "./_components/home/Analyzer.js";
import Hero from "./_components/home/Hero.js";
import Sources from "./_components/home/Sources.js";
import Bridge from "./_components/home/Bridge.js";
import Features from "./_components/home/Features.js";
import Maia from "./_components/home/Maia.js";

export default function Home() {
  useEffect(function () {
    Aos.init({
      duration: 1000,
      once: true,
    });
    return () => {
      Aos.refresh();
    };
  }, []);

  return (
    <>
      <div className="fixed inset-0">
        <AuroraBackground />
      </div>

      <main className="flex flex-col">
        <Hero />
        <Sources />
        <Bridge />
        <Analyzer />
        <Features />
        <Maia />
      </main>
    </>
  );
}

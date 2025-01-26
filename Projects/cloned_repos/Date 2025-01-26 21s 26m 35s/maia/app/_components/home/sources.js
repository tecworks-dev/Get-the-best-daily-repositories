"use client";

import Image from "next/image";
import { OrbitingCircles } from "../ui/OrbitingCircles";
import { cn } from "@/app/_lib/utils/uiUtils.js";

const Sources = () => {
  return (
    <section
      data-aos="fade-up"
      className="relative flex flex-col py-10 md:py-20 items-center justify-center overflow-hidden max-w-[1440px] mx-auto w-full"
    >
      <div className="relative flex items-center justify-center w-full md:min-h-screen sm:min-h-[700px] min-h-[500px]">
        <div className="relative flex flex-col gap-5 items-center justify-center max-w-72">
          <h1 className="text-2xl sm:text-3xl md:text-5xl font-oSemibold text-center text-white uppercase">
            MAIA Analyzer
          </h1>
        </div>

        {CIRCLE_SOURCES.map((source, index) => (
          <OrbitingCircles
            key={`first-${index}`}
            className="absolute size-[70px] border-none bg-transparent"
            radius={300}
            duration={50}
            delay={index * (50 / CIRCLE_SOURCES.length)}
          >
            <Image
              src={source.iconPath}
              width={70}
              height={70}
              alt={`source-${index + 1}`}
              className={cn(
                "object-contain w-full h-full max-w-20",
                `${source.isSupported ? "saturate-100" : "saturate-[0.2]"}`
              )}
            />
          </OrbitingCircles>
        ))}
      </div>
    </section>
  );
};

const CIRCLE_SOURCES = [
  {
    iconPath: "/assets/sources/source-1.png",
    isSupported: true,
  },
  {
    iconPath: "/assets/sources/source-2.png",
    isSupported: true,
  },
  {
    iconPath: "/assets/sources/source-3.png",
    isSupported: true,
  },
  {
    iconPath: "/assets/sources/source-4.png",
    isSupported: true,
  },
  {
    iconPath: "/assets/sources/source-5.png",
    isSupported: true,
  },
  {
    iconPath: "/assets/sources/source-6.png",
    isSupported: true,
  },
  {
    iconPath: "/assets/sources/source-7.png",
    isSupported: true,
  },
  {
    iconPath: "/assets/sources/source-8.png",
    isSupported: true,
  },
];

export default Sources;

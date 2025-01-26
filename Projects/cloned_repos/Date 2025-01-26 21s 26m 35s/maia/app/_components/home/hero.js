"use client";

import Image from "next/image";
import React from "react";
import { BorderButton } from "../ui/BorderButton";
import { OutlineButton } from "../ui/OutlineButton";
import { FlipWords } from "../ui/FlipWords.js";
import { BUY_LINK, GITHUB_LINK } from "@/config.js";
import { Socials } from "../ui/Socials.js";

export default function Hero() {
  const words = ["Twitter", "Bundles", "Creators", "Wallets", "News"];

  return (
    <section id="hero" className="relative flex items-center justify-center">
      <div className="z-10 bg-background-50 flex max-md:flex-col md:justify-beetween rounded-xl h-[94vh] m-6 w-full overflow-hidden">
        <div className="gap-[25px] h-full flex flex-col items-start justify-center md:pl-[85px] max-md:px-6 max-md:py-20 w-full max-md:order-1">
          <h1
            data-aos="fade-right"
            className="text-4xl md:text-5xl 2xl:text-[5rem] text-white"
          >
            I Scan <FlipWords words={words} /> <br /> to keep you ahead
          </h1>
          <div data-aos="fade-up" className="flex items-center gap-[19px]">
            <BorderButton
              onClick={() => window.open(BUY_LINK)}
              className={
                "font-sMedium text-sm flex items-center gap-1.5 h-[35px] w-24"
              }
              borderRadius="8px"
            >
              Buy MAIA
            </BorderButton>
            <OutlineButton onClick={() => window.open(GITHUB_LINK)}>
              DOCS
            </OutlineButton>
          </div>

          <Socials />
        </div>
        <Image
          data-aos="fade-up"
          src="/assets/hero.png"
          width={2000}
          height={2000}
          alt="hero image"
          className="md:h-full w-fit object-contain mr-auto"
        />
      </div>
      <Image
        src="/assets/decoration.png"
        width={700}
        height={700}
        alt="decoration"
        className="absolute z-0 -bottom-[400px] -left-[300px]"
        data-aos="fade-top-right"
      />
    </section>
  );
}

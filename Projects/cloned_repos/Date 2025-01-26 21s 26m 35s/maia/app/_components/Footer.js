"use client";

import { DISCORD_LINK, TELEGRAM_LINK, TWITTER_LINK } from "@/config.js";
import { Icons } from "./icon/icon";

const Footer = () => {
  return (
    <div className="flex flex-col gap-8 pb-[86px] px-6 pt-24">
      <div data-aos="fade-up" className="relative ">
        <h1 className="md:text-5xl text-3xl font-bold text-center text-white relative z-20 font-oSemibold">
          MAIA AI
        </h1>
        <div className="w-full h-full max-lg:h-[50vh] absolute top-12">
          <div className="absolute inset-x-0 top-0 bg-gradient-to-r from-transparent via-indigo-500 to-transparent h-[2px] w-3/4 blur-sm mx-auto" />
          <div className="absolute inset-x-0 top-0 bg-gradient-to-r from-transparent via-indigo-500 to-transparent h-px w-3/4 mx-auto" />
          <div className="absolute inset-x-0 top-0 bg-gradient-to-r from-transparent via-sky-500 to-transparent h-[5px] w-1/2 blur-sm mx-auto" />
          <div className="absolute inset-x-0 top-0 bg-gradient-to-r from-transparent via-sky-500 to-transparent h-px w-1/2 mx-auto" />
        </div>
      </div>

      <div
        data-aos="fade-up"
        className="flex flex-col items-center justify-center text-center w-full max-w-[472px] mx-auto text-white gap-9"
      >
        <p data-aos="fade-up">
          AI-powered guide, analyzing on-chain data and social trends to uncover
          the best token opportunities
        </p>
        <div className="flex items-center gap-6">
          <a
            href={TWITTER_LINK}
            className="text-foreground-50 hover:text-primary transition-all"
            target="_blank"
          >
            <Icons.xTwitter />
          </a>
          <a
            href={TELEGRAM_LINK}
            className="text-foreground-50 hover:text-primary transition-all"
            target="_blank"
          >
            <Icons.telegram />
          </a>
          <a
            href={DISCORD_LINK}
            className="text-foreground-50 hover:text-primary transition-all"
            target="_blank"
          >
            <Icons.discord />
          </a>
        </div>
      </div>
    </div>
  );
};

export default Footer;

import { TELEGRAM_LINK, TWITTER_LINK } from "@/config.js";
import { Icons } from "../icon/icon.js";

export const Socials = () => {
  return (
    <div
      data-aos="fade-up"
      className="flex items-start justify-start text-center w-full text-white gap-4"
    >
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
    </div>
  );
};

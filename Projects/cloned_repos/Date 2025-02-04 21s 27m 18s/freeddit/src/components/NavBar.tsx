import Search from "./Search";
import { useEffect, useState } from "react";
import Link from "next/link";
import NavMenu from "./NavMenu";
import { useRouter } from "next/router";
import SortMenu from "./SortMenu";
import { useScroll } from "../hooks/useScroll";
import { usePlausible } from "next-plausible";
import { useMainContext } from "../MainContext";
import FilterMenu from "./FilterMenu";
import { FaTelegram } from "react-icons/fa";
import { FaXTwitter } from "react-icons/fa6";

const NavBar = ({ toggleSideNav = 0 }) => {
  const context: any = useMainContext();
  const { setForceRefresh } = context;
  const [hidden, setHidden] = useState(false);
  const [allowHide, setallowHide] = useState(true);
  const router = useRouter();
  const plausible = usePlausible();
  const { scrollY, scrollX, scrollDirection } = useScroll();

  useEffect(() => {
    if (allowHide && !context?.loading) {
      if (scrollDirection === "down" || !scrollY) {
        setHidden(false);
      } else if (scrollY > 300 && scrollDirection === "up" && !hidden) {
        setHidden(true);
      } else if (scrollY <= 300) {
        setHidden(false);
      }
    } else {
      hidden && setHidden(false);
    }
  }, [scrollDirection, allowHide, scrollY]);

  const forceShow = () => {
    if (hidden) {
      setHidden(false);
    }
  };

  useEffect(() => {
    forceShow();
    if (
      router.query?.slug?.[1] === "comments" ||
      router.pathname.includes("/about") ||
      router.pathname.includes("/subreddits")
    ) {
      setallowHide(false);
    } else {
      setallowHide(true);
    }
    return () => {
      //setallowHide(true);
    };
  }, [router]);

  const homeClick = () => {
    router?.route === "/" && setForceRefresh((p) => p + 1);
  };

  return (
    <>
      <header
        className={
          `${hidden ? "-translate-y-full" : ""}` +
          " z-50 fixed top-0 transition duration-500 ease-in-out transform h-14 w-screen "
        }
      >
        <nav className="flex flex-row items-center flex-grow h-full shadow-lg bg-th-background2 md:justify-between ">
          <div className="flex flex-row items-center justify-start h-full mr-2">
            <Link href="/" passHref>
              <a>
                <h1
                  className="ml-2 text-2xl align-middle cursor-pointer select-none"
                  onClick={homeClick}
                >
                  {"freeddit"}
                </h1>
              </a>
            </Link>
          </div>
          <div className="w-full h-full py-2 max-w-7xl md:block">
            <Search id={"subreddit search main"} />
          </div>
          <div className="flex items-center ml-4 space-x-2">
            <a
              href="https://t.me/freereddit"
              target="_blank"
              rel="noreferrer"
              className="hover:cursor-pointer"
            >
              <FaTelegram className="w-6 h-6 transition-all hover:scale-110" />
            </a>
            <a
              href="https://x.com/freeddit"
              target="_blank"
              rel="noreferrer"
              className="hover:cursor-pointer"
            >
              <FaXTwitter className="w-6 h-6 transition-all hover:scale-110" />
            </a>
          </div>
          <div className="flex flex-row items-center justify-end h-full py-2 ml-auto mr-2 space-x-1 md:ml-2">
            <div className="w-20 h-full">
              <SortMenu hide={hidden} />
            </div>
            <div
              className="flex flex-row items-center w-10 h-full mr-2 "
              onClick={() => plausible("filters")}
            >
              <FilterMenu hide={hidden} />
            </div>
            <div
              className={
                "hidden w-20 h-full border hover:border-th-border border-transparent rounded-md md:block"
              }
            >
              <Link href="/about">
                <a className="flex items-center justify-center w-full h-full">
                  About
                </a>
              </Link>
            </div>

            <div
              className="flex flex-row items-center w-10 h-full mr-2 "
              onClick={() => plausible("options")}
            >
              <NavMenu hide={hidden} />
            </div>
          </div>
        </nav>
      </header>
      <div
        onMouseOver={(e) => forceShow()}
        className="fixed top-0 z-40 w-full bg-transparent h-14 opacity-10 "
      ></div>
    </>
  );
};

export default NavBar;

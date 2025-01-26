import { ThemeToggle } from "@/components/ThemeToggle";
import Link from "next/link";
import { BsGithub } from "react-icons/bs";

const Header = () => {
  return (
    <header className="border-b border-gray-200 dark:border-gray-800">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center">
            <Link href="/" className="flex items-center space-x-2">
              <span className="text-xl font-bold text-gray-900 dark:text-white">
                Image URL Converter
              </span>
            </Link>
          </div>

          <div className="flex items-center space-x-4">
            <Link
              href="https://github.com/weijunext/image-url-converter"
              target="_blank"
              className="text-gray-700 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <BsGithub className="h-6 w-6" />
            </Link>
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;

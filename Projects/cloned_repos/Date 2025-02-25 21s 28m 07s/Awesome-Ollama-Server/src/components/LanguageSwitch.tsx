import { useState } from 'react';
import { Locale, locales } from '@/config';
import { GlobeAltIcon } from '@heroicons/react/24/outline';
import { useRouter, usePathname } from '@/navigation';

interface LanguageSwitchProps {
  currentLocale: Locale;
}

export function LanguageSwitch({ currentLocale }: LanguageSwitchProps) {
  const [isOpen, setIsOpen] = useState(false);
  const router = useRouter();
  const pathname = usePathname();

  const handleLanguageChange = (locale: Locale) => {
    setIsOpen(false);
    router.replace(pathname, { locale });
  };

  const languageNames: Record<Locale, string> = {
    en: 'English',
    zh: '中文',
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-1 px-3 py-2 rounded-md text-gray-300 hover:text-white
          hover:bg-gray-700 transition-colors duration-200"
      >
        <GlobeAltIcon className="h-5 w-5" />
        <span>{languageNames[currentLocale]}</span>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-gray-800 ring-1 ring-black ring-opacity-5">
          <div className="py-1" role="menu" aria-orientation="vertical" aria-labelledby="language-menu">
            {locales.map((locale) => (
              <button
                key={locale}
                onClick={() => handleLanguageChange(locale)}
                className={`block w-full text-left px-4 py-2 text-sm
                  ${currentLocale === locale
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  } transition-colors duration-200`}
                role="menuitem"
              >
                {languageNames[locale]}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 
'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';

enum MenuState {
  CLOSED,
  OPEN,
  BACK,
}

const topBarVariants = {
  closed: {
    rotate: 0,
    translateY: 0,
    width: '16px',
  },
  open: {
    rotate: 45,
    translateY: 4,
    width: '16px',
  },
  back: {
    rotate: 45,
    translateY: 7,
    width: '10px',
  },
};

const bottomBarVariants = {
  closed: {
    rotate: 0,
    translateY: 0,
    width: '16px',
  },
  open: {
    rotate: -45,
    translateY: -4,
    width: '16px',
  },
  back: {
    rotate: -45,
    translateY: -7,
    width: '10px',
  },
};

interface MainMenuProps {
  isBackVisible: boolean;
  onBack: any;
}

export function MainMenu({ isBackVisible, onBack }: MainMenuProps) {
  const pathname = usePathname();
  const defaultMenuState =
    isBackVisible || pathname.startsWith('/p/')
      ? MenuState.BACK
      : MenuState.CLOSED;

  const [menuState, setMenuState] = useState<MenuState>(defaultMenuState);

  useEffect(() => {
    setMenuState(isBackVisible ? MenuState.BACK : MenuState.CLOSED);
  }, [isBackVisible]);

  const handleClick = () => {
    switch (menuState) {
      case MenuState.CLOSED:
        setMenuState(MenuState.OPEN);
        break;
      case MenuState.OPEN:
        setMenuState(MenuState.CLOSED);
        break;
      case MenuState.BACK:
        onBack();
        break;
    }
  };

  const getVariant = (state: MenuState) => {
    switch (state) {
      case MenuState.OPEN:
        return 'open';
      case MenuState.BACK:
        return 'back';
      default:
        return 'closed';
    }
  };

  return (
    <div className="relative flex items-center">
      <button
        className="p-2 z-20 relative size-12 flex items-center justify-center"
        onClick={handleClick}
        aria-expanded={menuState === MenuState.OPEN}
        aria-label={menuState === MenuState.BACK ? 'Back' : 'Menu'}
      >
        <span className="sr-only">
          {menuState === MenuState.BACK ? 'Back' : 'Menu'}
        </span>

        <div className="size-4 relative flex items-center justify-center">
          {/* Top bar */}
          <motion.span
            className="absolute left-0 top-[4px] w-4 h-[2px] bg-black origin-center"
            variants={topBarVariants}
            initial={getVariant(defaultMenuState)}
            animate={getVariant(menuState)}
            transition={{ duration: 0.3, ease: [0.68, -0.55, 0.265, 1.55] }}
          />
          {/* Bottom bar */}
          <motion.span
            className="absolute left-0 top-[12px] w-4 h-[2px] bg-black origin-center"
            variants={bottomBarVariants}
            initial={getVariant(defaultMenuState)}
            animate={getVariant(menuState)}
            transition={{ duration: 0.3, ease: [0.68, -0.55, 0.265, 1.55] }}
          />
        </div>
      </button>

      <AnimatePresence>
        {menuState === MenuState.OPEN && (
          <motion.nav
            id="main-menu"
            className="absolute left-1/2 top-[20%] -translate-y-1/2 ml-8"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            <ul className="flex items-center space-x-2">
              {['HELP', 'TERMS', 'PRIVACY', 'ABOUT'].map((item, index) => (
                <motion.li
                  key={item}
                  className={`bg-white px-3 py-1 rounded ${
                    item === 'PRIVACY' ? 'hidden sm:block' : ''
                  }`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Link
                    href={`/${item.toLowerCase()}`}
                    className="text-sm font-mono hover:opacity-70 transition-opacity whitespace-nowrap"
                  >
                    {item}
                  </Link>
                </motion.li>
              ))}
            </ul>
          </motion.nav>
        )}
      </AnimatePresence>
    </div>
  );
}

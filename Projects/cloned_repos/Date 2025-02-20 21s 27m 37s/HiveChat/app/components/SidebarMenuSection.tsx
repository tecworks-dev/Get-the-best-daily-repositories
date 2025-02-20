import React, { useState, useRef, FC, ReactNode } from 'react';
import ChevronDown from "@/app/images/chevrondown.svg";
import clsx from 'clsx';

interface MenuSectionProps {
  title: string;
  icon: ReactNode;
  children: ReactNode;
  defaultExpanded?: boolean;
}
const MenuSection: FC<MenuSectionProps> = ({
  title,
  icon,
  children,
  defaultExpanded = false
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(defaultExpanded);
  const contentRef = useRef<HTMLDivElement>(null);

  return (
    <div className="mb-2">
      <div
        className="flex rounded-xl mr-4 items-center justify-between px-2 py-2 cursor-pointer hover:bg-gray-200 transition-colors duration-200"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center">
          {icon}
          <span className="text-gray-800 ml-1 text-sm whitespace-nowrap">{title}</span>
        </div>
        <ChevronDown width={16} height={16} alt='ChevronDown'
          className={`w-5 h-5 text-gray-500 opacity-20 transition-transform duration-300 ease-in-out ${isExpanded ? '' : 'transform -rotate-90'
            }`}
        />
      </div>
      <div
        className={clsx(
          "transition-all duration-300 ease-in-out overflow-hidden",
          isExpanded ? "max-h-82" : "max-h-0"
        )}
      >
        <div className="transform transition-transform duration-300 ease-in-out">
          {children}
        </div>
      </div>
    </div>
  );
};

export default MenuSection
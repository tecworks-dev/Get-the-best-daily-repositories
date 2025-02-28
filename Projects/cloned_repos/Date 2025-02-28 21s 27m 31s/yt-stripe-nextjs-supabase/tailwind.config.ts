import type { Config } from "tailwindcss";

export default {
  darkMode: 'media',
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#A78BFA', // Violet-400: Soft lavender
          light: '#C4B5FD',   // Violet-300: Light lavender
          dark: '#8B5CF6',    // Violet-500: Medium lavender
        },
        danger: {
          DEFAULT: '#DC2626', // Red-600: Clear red
          light: '#F87171',   // Red-400: Soft red
          dark: '#B91C1C',    // Red-700: Deep red
        },
        neutral: {
          DEFAULT: '#F8FAFC', // Slate-50: Crisp light
          dark: '#1E293B',    // Slate-800: Rich dark
          darker: '#0F172A',  // Slate-900: Deep dark
        },
        text: {
          DEFAULT: '#0F172A', // Slate-900: Sharp text
          light: '#64748B',   // Slate-500: Soft text
          dark: '#F8FAFC',    // Slate-50: Light text
        },
        surface: {
          light: '#FFFFFF',   // Pure white
          dark: '#1E293B',    // Slate-800: Rich surface
        },
        accent: {
          DEFAULT: '#38BDF8', // Sky-400: Fresh blue
          light: '#7DD3FC',   // Sky-300: Soft blue
          dark: '#0EA5E9',    // Sky-500: Deep blue
        }
      },
      boxShadow: {
        'subtle': '0 1px 3px rgba(0,0,0,0.05)',
        'hover': '0 4px 6px -1px rgba(139, 92, 246, 0.1), 0 2px 4px -1px rgba(139, 92, 246, 0.06)', // Softer violet shadow
      }
    },
  },
  plugins: [],
} satisfies Config;

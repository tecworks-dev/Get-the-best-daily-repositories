import type { Config } from 'tailwindcss'

const { nextui } = require('@nextui-org/react')
const typography = require('@tailwindcss/typography')
const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './framework/**/*.{js,ts,jsx,tsx,mdx}',
    './node_modules/@nextui-org/theme/dist/**/*.{js,ts,jsx,tsx}'
  ],
  theme: {
    extend: {
      textColor: {
        slate: {
          50: 'var(--ai-create-color-theme-50)',
          100: 'var(--ai-create-color-theme-100)',
          200: 'var(--ai-create-color-theme-200)',
          300: 'var(--ai-create-color-theme-300)',
          400: 'var(--ai-create-color-theme-400)',
          500: 'var(--ai-create-color-theme-500)',
          600: 'var(--ai-create-color-theme-600)',
          700: 'var(--ai-create-color-theme-700)',
          800: 'var(--ai-create-color-theme-800)',
          900: 'var(--ai-create-color-theme-900)',
          950: 'var(--ai-create-color-theme-950)'
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
          950: '#030712'
        }
      },
      colors: {
        primary: '#008080',
        inherit: 'inherit',
        current: 'currentColor',
        transparent: 'transparent',
        black: '#000',
        white: '#fff',
        slate: {
          50: 'var(--ai-create-color-theme-50)',
          100: 'var(--ai-create-color-theme-100)',
          200: 'var(--ai-create-color-theme-200)',
          300: 'var(--ai-create-color-theme-300)',
          400: 'var(--ai-create-color-theme-400)',
          500: 'var(--ai-create-color-theme-500)',
          600: 'var(--ai-create-color-theme-600)',
          700: 'var(--ai-create-color-theme-700)',
          800: 'var(--ai-create-color-theme-800)',
          900: 'var(--ai-create-color-theme-900)',
          950: 'var(--ai-create-color-theme-950)'
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
          950: '#030712'
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))'
      }
    }
  },
  darkMode: 'class',
  plugins: [
    nextui({
      themes: {
        light: {
          colors: {
            primary: {
              DEFAULT: '#008080',
              200: '#03e0e0',
              300: '#01c4c4'
            },
            secondary: {
              DEFAULT: '#af46ef',
              200: '#ffe0ff'
            },
            fuchsia: {
              '50': '#fdf4ff',
              '100': '#fae8ff',
              '200': '#f5d0fe',
              '300': '#f0abfc',
              '400': '#e879f9',
              '500': '#d946ef',
              '600': '#c026d3',
              '700': '#a21caf',
              '800': '#86198f',
              '900': '#701a75',
              '950': '#4a044e'
            },
            black: {
              DEFAULT: '#2d2d2d',
              300: '#454545'
            },
            gray: {
              DEFAULT: '#e8e8e8'
            }
          }
        }
      }
    }),
    typography()
  ]
}
export default config

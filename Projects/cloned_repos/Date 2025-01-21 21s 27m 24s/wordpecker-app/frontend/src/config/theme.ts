import { theme as baseTheme } from '@chakra-ui/react';

export const theme = {
  ...baseTheme,
  styles: {
    global: {
      body: {
        bg: 'gray.50',
        color: 'gray.900',
      },
    },
  },
  colors: {
    ...baseTheme.colors,
    brand: {
      50: '#f0f9ff',
      100: '#e0f2fe',
      200: '#bae6fd',
      300: '#7dd3fc',
      400: '#38bdf8',
      500: '#0ea5e9',
      600: '#0284c7',
      700: '#0369a1',
      800: '#075985',
      900: '#0c4a6e',
    },
  },
  components: {
    ...baseTheme.components,
    Button: {
      defaultProps: {
        colorScheme: 'brand',
      },
    },
  },
}; 
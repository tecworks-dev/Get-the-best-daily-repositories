import { extendTheme, type ThemeConfig } from '@chakra-ui/react';

const config: ThemeConfig = {
  initialColorMode: 'dark',
  useSystemColorMode: false,
};

const theme = extendTheme({
  config,
  styles: {
    global: {
      body: {
        bg: 'slate.900',
        color: 'white',
      },
    },
  },
  colors: {
    slate: {
      900: '#0f172a', // background
      800: '#1e293b', // cards/containers
      700: '#334155', // interactive elements
    },
    brand: {
      400: '#4ade80', // primary action/brand color
    },
    status: {
      xp: '#facc15', // yellow-400
      streak: '#fb923c', // orange-400
      heart: '#ef4444', // red-500
    }
  },
  components: {
    Container: {
      baseStyle: {
        maxW: 'container.xl',
        px: 6,
      },
    },
    Card: {
      baseStyle: {
        container: {
          bg: 'slate.800',
          rounded: 'xl',
          p: 4,
        },
      },
    },
    Button: {
      baseStyle: {
        rounded: 'lg',
        fontWeight: 'medium',
      },
      variants: {
        solid: {
          bg: 'brand.400',
          color: 'slate.900',
          _hover: {
            bg: 'green.300',
          },
        },
        ghost: {
          _hover: {
            bg: 'slate.700',
          },
        },
      },
    },
    Badge: {
      baseStyle: {
        rounded: 'full',
      },
    },
    Progress: {
      baseStyle: {
        track: {
          bg: 'slate.700',
        },
        filledTrack: {
          bg: 'brand.400',
          transition: 'all 0.3s',
        },
      },
    },
  },
  layerStyles: {
    card: {
      bg: 'slate.800',
      rounded: 'xl',
      p: 4,
      shadow: 'md',
    },
    interactive: {
      cursor: 'pointer',
      transition: 'all 0.2s',
      _hover: {
        bg: 'slate.700',
      },
    },
  },
  textStyles: {
    h1: {
      fontSize: ['2xl', '3xl'],
      fontWeight: 'bold',
      lineHeight: 'shorter',
      mb: 4,
    },
    h2: {
      fontSize: ['xl', '2xl'],
      fontWeight: 'semibold',
      lineHeight: 'short',
      mb: 3,
    },
  },
});

export default theme; 
import { Box, Container } from '@chakra-ui/react';
import { ReactNode } from 'react';
import { Header } from './Header';
import { Footer } from './Footer';

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  return (
    <Box minH="100vh" w="100%" bg="slate.900" display="flex" flexDirection="column">
      <Header />
      <Container py={6} flex="1">
        {children}
      </Container>
      <Footer />
    </Box>
  );
}; 
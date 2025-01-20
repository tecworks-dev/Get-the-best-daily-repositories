import { Box, Container, Text, Flex } from '@chakra-ui/react';

export const Footer = () => {
  return (
    <Box as="footer" w="100%" bg="slate.800" borderTop="1px" borderColor="slate.700" py={4}>
      <Container>
        <Flex justifyContent="space-between" alignItems="center">
          <Text color="gray.400">
            © {new Date().getFullYear()} List Generator. All rights reserved.
          </Text>
          <Flex gap={4}>
            <Text as="a" href="#" color="gray.400" _hover={{ color: 'brand.400' }}>
              Privacy Policy
            </Text>
            <Text as="a" href="#" color="gray.400" _hover={{ color: 'brand.400' }}>
              Terms of Service
            </Text>
          </Flex>
        </Flex>
      </Container>
    </Box>
  );
}; 
import { useState } from 'react';
import {
  Box,
  Button,
  Container,
  FormControl,
  FormLabel,
  Heading,
  Input,
  Stack,
  Text,
  useToast,
  Icon,
  VStack,
  Flex,
} from '@chakra-ui/react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { FaFeatherAlt } from 'react-icons/fa';
import { useAuth } from '../contexts/AuthContext';

export const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const toast = useToast();

  const from = location.state?.from?.pathname || '/lists';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await signIn(email, password);
      navigate(from, { replace: true });
    } catch (error) {
      toast({
        title: 'Error signing in',
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxW="container.sm" py={8}>
      <VStack spacing={8}>
        <Flex direction="column" align="center">
          <Icon 
            as={FaFeatherAlt} 
            boxSize={12} 
            color="orange.400" 
            transform="rotate(-45deg)"
            mb={4}
          />
          <Heading 
            as="h1" 
            size="xl"
            bgGradient="linear(to-r, orange.400, brand.400)"
            bgClip="text"
            textAlign="center"
          >
            Welcome back to WordPecker
          </Heading>
          <Text color="gray.400" fontSize="lg" mt={2}>
            Sign in to continue your vocabulary journey
          </Text>
        </Flex>

        <Box
          as="form"
          onSubmit={handleSubmit}
          bg="slate.800"
          p={8}
          borderRadius="xl"
          borderWidth="1px"
          borderColor="slate.700"
          w="full"
        >
          <Stack spacing={4}>
            <FormControl isRequired>
              <FormLabel color="gray.300">Email</FormLabel>
              <Input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                bg="slate.700"
                borderColor="slate.600"
                _hover={{ borderColor: 'slate.500' }}
                _focus={{ borderColor: 'brand.400', boxShadow: 'none' }}
                color="white"
                size="lg"
              />
            </FormControl>
            <FormControl isRequired>
              <FormLabel color="gray.300">Password</FormLabel>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                bg="slate.700"
                borderColor="slate.600"
                _hover={{ borderColor: 'slate.500' }}
                _focus={{ borderColor: 'brand.400', boxShadow: 'none' }}
                color="white"
                size="lg"
              />
            </FormControl>
            <Button
              type="submit"
              colorScheme="orange"
              size="lg"
              isLoading={isLoading}
              loadingText="Signing in..."
              _hover={{
                transform: 'translateY(-2px)',
                shadow: 'lg'
              }}
              transition="all 0.2s"
            >
              Sign In
            </Button>
          </Stack>
        </Box>

        <Text color="gray.400">
          Don't have an account?{' '}
          <Link to="/register">
            <Button
              variant="link"
              color="brand.400"
              _hover={{ color: 'brand.300' }}
            >
              Sign up
            </Button>
          </Link>
        </Text>
      </VStack>
    </Container>
  );
}; 
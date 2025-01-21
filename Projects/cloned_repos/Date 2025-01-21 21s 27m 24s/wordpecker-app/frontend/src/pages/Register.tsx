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
import { Link, useNavigate } from 'react-router-dom';
import { FaFeatherAlt } from 'react-icons/fa';
import { useAuth } from '../contexts/AuthContext';

export const Register = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { signUp } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (password !== confirmPassword) {
      toast({
        title: 'Passwords do not match',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsLoading(true);

    try {
      await signUp(email, password);
      toast({
        title: 'Account created successfully!',
        description: 'Please check your email to confirm your account.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      navigate('/login');
    } catch (error) {
      toast({
        title: 'Error creating account',
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
            Join WordPecker
          </Heading>
          <Text color="gray.400" fontSize="lg" mt={2}>
            Create an account to start your vocabulary journey
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
            <FormControl isRequired>
              <FormLabel color="gray.300">Confirm Password</FormLabel>
              <Input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
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
              colorScheme="green"
              size="lg"
              isLoading={isLoading}
              loadingText="Creating account..."
              _hover={{
                transform: 'translateY(-2px)',
                shadow: 'lg'
              }}
              transition="all 0.2s"
            >
              Create Account
            </Button>
          </Stack>
        </Box>

        <Text color="gray.400">
          Already have an account?{' '}
          <Link to="/login">
            <Button
              variant="link"
              color="brand.400"
              _hover={{ color: 'brand.300' }}
            >
              Sign in
            </Button>
          </Link>
        </Text>
      </VStack>
    </Container>
  );
}; 
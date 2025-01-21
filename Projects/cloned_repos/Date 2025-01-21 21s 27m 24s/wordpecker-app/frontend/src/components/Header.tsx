import { Box, Container, Flex, Button, Icon, Text, Menu, MenuButton, MenuList, MenuItem, Avatar } from '@chakra-ui/react';
import { Link, useNavigate } from 'react-router-dom';
import { GiTreeBranch } from 'react-icons/gi';
import { FaHome, FaFeatherAlt, FaSignInAlt, FaUserPlus, FaSignOutAlt, FaUser } from 'react-icons/fa';
import { useAuth } from '../contexts/AuthContext';

export const Header = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <Box as="nav" w="100%" bg="slate.800" boxShadow="lg" position="sticky" top={0} zIndex={10}>
      <Container maxW="container.xl">
        <Flex h="16" alignItems="center" justifyContent="space-between">
          <Flex gap={6} align="center">
            <Link to="/">
              <Button 
                variant="ghost" 
                fontWeight="bold"
                leftIcon={<Icon as={FaHome} />}
                _hover={{
                  transform: 'translateY(-2px)',
                  color: 'green.400'
                }}
                transition="all 0.2s"
              >
                Home
              </Button>
            </Link>
            {user && (
              <Link to="/lists">
                <Button 
                  variant="ghost"
                  leftIcon={<Icon as={GiTreeBranch} color="green.400" />}
                  _hover={{
                    transform: 'translateY(-2px)',
                    color: 'green.400'
                  }}
                  transition="all 0.2s"
                >
                  <Flex align="center" gap={1}>
                    <Text>My Trees</Text>
                    <Icon 
                      as={FaFeatherAlt} 
                      color="orange.400" 
                      transform="rotate(-45deg)"
                      boxSize={3}
                      ml={-1}
                      mt={-2}
                    />
                  </Flex>
                </Button>
              </Link>
            )}
          </Flex>
          <Flex gap={4} align="center">
            {user ? (
              <Menu>
                <MenuButton
                  as={Button}
                  variant="ghost"
                  _hover={{
                    transform: 'translateY(-2px)',
                    color: 'green.400'
                  }}
                  transition="all 0.2s"
                >
                  <Flex align="center" gap={2}>
                    <Avatar 
                      size="sm" 
                      name={user.email} 
                      bg="green.500"
                      icon={<Icon as={FaUser} color="white" />}
                    />
                    <Text color="gray.300">{user.email}</Text>
                  </Flex>
                </MenuButton>
                <MenuList bg="slate.700" borderColor="slate.600">
                  <MenuItem
                    icon={<Icon as={FaSignOutAlt} />}
                    onClick={handleSignOut}
                    _hover={{
                      bg: 'slate.600',
                      color: 'orange.400'
                    }}
                  >
                    Sign Out
                  </MenuItem>
                </MenuList>
              </Menu>
            ) : (
              <>
                <Link to="/login">
                  <Button 
                    variant="ghost"
                    leftIcon={<Icon as={FaSignInAlt} />}
                    _hover={{
                      transform: 'translateY(-2px)',
                      color: 'orange.400'
                    }}
                    transition="all 0.2s"
                  >
                    Login
                  </Button>
                </Link>
                <Link to="/register">
                  <Button 
                    variant="solid"
                    colorScheme="green"
                    leftIcon={<Icon as={FaUserPlus} />}
                    _hover={{
                      transform: 'translateY(-2px)'
                    }}
                    transition="all 0.2s"
                  >
                    Sign Up
                  </Button>
                </Link>
              </>
            )}
          </Flex>
        </Flex>
      </Container>
    </Box>
  );
}; 
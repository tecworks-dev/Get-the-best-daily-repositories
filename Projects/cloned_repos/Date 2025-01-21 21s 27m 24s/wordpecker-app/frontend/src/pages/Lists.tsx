import {
  Box,
  Button,
  Container,
  Flex,
  Heading,
  Icon,
  Text,
  useDisclosure,
  keyframes,
  VStack,
  Badge,
  SimpleGrid,
  useToast,
  Spinner,
  Center
} from '@chakra-ui/react';
import { motion } from 'framer-motion';
import { GiTreeBranch } from 'react-icons/gi';
import { FaPlus, FaGamepad, FaBook, FaFeatherAlt, FaEye } from 'react-icons/fa';
import { CreateListModal } from '../components/CreateListModal';
import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { WordList } from '../types';
import { apiService } from '../services/api';

const sparkle = keyframes`
  0% { transform: scale(1) rotate(0deg); }
  25% { transform: scale(1.1) rotate(-5deg); }
  50% { transform: scale(1) rotate(0deg); }
  75% { transform: scale(1.1) rotate(5deg); }
  100% { transform: scale(1) rotate(0deg); }
`;

const float = keyframes`
  0% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0px); }
`;

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

export const Lists = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const toast = useToast();
  const [lists, setLists] = useState<WordList[]>([]);
  const [wordCounts, setWordCounts] = useState<{ [key: string]: number }>({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchLists = async () => {
      try {
        const listsData = await apiService.getLists();
        setLists(listsData);
        
        // Fetch word counts for each list
        const counts: { [key: string]: number } = {};
        await Promise.all(
          listsData.map(async (list) => {
            const words = await apiService.getWords(list.id);
            counts[list.id] = words.length;
          })
        );
        setWordCounts(counts);
      } catch (error) {
        console.error('Error fetching lists:', error);
        toast({
          title: 'Error fetching lists',
          description: 'Please try again later',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchLists();
  }, [toast]);

  const handleCreateList = async (name: string, description: string, context: string) => {
    try {
      const newList = await apiService.createList({ name, description, context });
      setLists(prevLists => [newList, ...prevLists]);
      setWordCounts(prev => ({ ...prev, [newList.id]: 0 }));
      toast({
        title: 'List created successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      onClose();
    } catch (error) {
      console.error('Error creating list:', error);
      toast({
        title: 'Error creating list',
        description: 'Please try again later',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  if (isLoading) {
    return (
      <Center h="calc(100vh - 64px)">
        <Spinner size="xl" color="green.400" thickness="4px" />
      </Center>
    );
  }

  return (
    <Container maxW="container.xl" py={8} px={{ base: 4, md: 8 }}>
      <VStack spacing={8} align="stretch">
        <Flex 
          justify="space-between" 
          align={{ base: 'flex-start', md: 'center' }}
          direction={{ base: 'column', md: 'row' }}
          gap={4}
        >
          <Box>
            <Heading 
              as="h1" 
              size="2xl"
              bgGradient="linear(to-r, green.400, brand.400)"
              bgClip="text"
              display="flex"
              alignItems="center"
              gap={3}
            >
              <Flex align="center" gap={2}>
                <Icon 
                  as={GiTreeBranch} 
                  boxSize={10} 
                  color="green.400"
                  animation={`${sparkle} 3s ease infinite`}
                />
                <Icon 
                  as={FaFeatherAlt} 
                  boxSize={8} 
                  color="orange.400"
                  transform="rotate(-45deg)"
                  animation={`${sparkle} 2s ease infinite`}
                  ml={-4}
                  mt={-4}
                />
              </Flex>
              My Word Trees
            </Heading>
            <Text mt={2} color="gray.400" fontSize="lg">
              Plant and grow your vocabulary with WordPecker! 🌱
            </Text>
          </Box>
          <Button
            leftIcon={<FaPlus />}
            colorScheme="green"
            size="lg"
            onClick={onOpen}
            _hover={{
              transform: 'translateY(-2px)',
              animation: `${sparkle} 1s ease infinite`
            }}
            transition="all 0.2s"
            w={{ base: 'full', md: 'auto' }}
          >
            Plant New Tree
          </Button>
        </Flex>

        <SimpleGrid
          columns={{ base: 1, md: 2, lg: 3 }}
          spacing={6}
          as={motion.div}
          variants={container}
          initial="hidden"
          animate="show"
        >
          {lists.map((list) => {
            const wordCount = wordCounts[list.id] || 0;
            
            return (
              <Box
                key={list.id}
                as={motion.div}
                variants={item}
                whileHover={{ 
                  y: -5,
                  transition: { duration: 0.2 }
                }}
              >
                <Box
                  layerStyle="card"
                  h="full"
                  position="relative"
                  overflow="hidden"
                  borderWidth="1px"
                  borderColor="green.800"
                  _hover={{
                    borderColor: "green.600",
                    shadow: "0 0 20px rgba(72, 187, 120, 0.3)"
                  }}
                  transition="all 0.3s"
                  p={6}
                  borderRadius="xl"
                >
                  <Flex direction="column" h="full" gap={4}>
                    <Box>
                      <Flex align="center" gap={2} mb={3}>
                        <Icon as={GiTreeBranch} color="green.400" boxSize={5} />
                        <Text 
                          fontWeight="bold"
                          fontSize="xl"
                          color="white"
                          noOfLines={1}
                        >
                          {list.name}
                        </Text>
                      </Flex>
                      <Text 
                        color="gray.400" 
                        fontSize="md"
                        noOfLines={2}
                      >
                        {list.description}
                      </Text>
                    </Box>

                    <Box 
                      position="absolute"
                      top={4}
                      right={4}
                      animation={wordCount > 0 ? `${float} 3s ease-in-out infinite` : undefined}
                    >
                      <Badge 
                        colorScheme={wordCount > 0 ? "green" : "gray"}
                        p={2}
                        borderRadius="full"
                        fontSize="sm"
                      >
                        {wordCount} {wordCount === 1 ? 'word' : 'words'}
                      </Badge>
                    </Box>
                    
                    <Flex 
                      mt="auto" 
                      gap={2} 
                      flexWrap="wrap"
                      direction={{ base: 'column', sm: 'row' }}
                    >
                      <Link to={`/learn/${list.id}`} style={{ flex: 1 }}>
                        <Button 
                          w="full"
                          variant="ghost"
                          colorScheme="green"
                          leftIcon={<Icon as={FaBook} boxSize={5} />}
                          _hover={{
                            transform: 'translateY(-2px)',
                            shadow: 'md'
                          }}
                          transition="all 0.2s"
                          isDisabled={wordCount === 0}
                          size="md"
                        >
                          Learn
                        </Button>
                      </Link>
                      <Link to={`/quiz/${list.id}`} style={{ flex: 1 }}>
                        <Button 
                          w="full"
                          variant="ghost"
                          colorScheme="orange"
                          leftIcon={<Icon as={FaGamepad} />}
                          _hover={{
                            transform: 'translateY(-2px)',
                            shadow: 'md'
                          }}
                          transition="all 0.2s"
                          isDisabled={wordCount === 0}
                          size="md"
                        >
                          Quiz
                        </Button>
                      </Link>
                      <Link to={`/lists/${list.id}`} style={{ flex: 1 }}>
                        <Button 
                          w="full"
                          variant="ghost"
                          colorScheme="blue"
                          leftIcon={<Icon as={FaEye} />}
                          _hover={{
                            transform: 'translateY(-2px)',
                            shadow: 'md'
                          }}
                          transition="all 0.2s"
                          size="md"
                        >
                          View
                        </Button>
                      </Link>
                    </Flex>
                  </Flex>
                </Box>
              </Box>
            );
          })}
        </SimpleGrid>
      </VStack>

      <CreateListModal
        isOpen={isOpen}
        onClose={onClose}
        onCreateList={handleCreateList}
      />
    </Container>
  );
}; 
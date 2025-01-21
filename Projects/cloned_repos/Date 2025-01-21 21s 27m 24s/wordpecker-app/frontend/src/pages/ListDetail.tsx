import { 
  Box, 
  Button, 
  Text, 
  Flex, 
  keyframes, 
  IconButton, 
  useDisclosure, 
  Container, 
  Heading, 
  Icon,
  useToast,
  Spinner,
  Center
} from '@chakra-ui/react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Word, WordList } from '../types';
import { ArrowBackIcon, DeleteIcon } from '@chakra-ui/icons';
import { FaGraduationCap, FaGamepad, FaPlus } from 'react-icons/fa';
import { GiTreeBranch } from 'react-icons/gi';
import { AddWordModal } from '../components/AddWordModal';
import { apiService } from '../services/api';

// Animation keyframes
const sparkle = keyframes`
  0% { transform: scale(1) rotate(0deg); }
  25% { transform: scale(1.1) rotate(-5deg); }
  50% { transform: scale(1) rotate(0deg); }
  75% { transform: scale(1.1) rotate(5deg); }
  100% { transform: scale(1) rotate(0deg); }
`;

// Dynamic color generator
const generateColor = (word: string) => {
  // Generate a hue based on the word's characters
  const hue = word.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % 360;
  // Use fixed saturation and lightness for consistency
  return `hsl(${hue}, 70%, 25%)`;
};

// Generate hover color (slightly lighter version)
const generateHoverColor = (word: string) => {
  const hue = word.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % 360;
  return `hsl(${hue}, 70%, 30%)`;
};

const MotionBox = motion(Box);

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

export const ListDetail = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  const [list, setList] = useState<WordList | null>(null);
  const [words, setWords] = useState<Word[]>([]);
  const [selectedWord, setSelectedWord] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchListDetails = async () => {
      if (!id) return;
      
      try {
        // Fetch list details
        const listData = await apiService.getList(id);
        setList(listData);

        // Fetch words
        const wordsData = await apiService.getWords(id);
        setWords(wordsData);
      } catch (error) {
        console.error('Error fetching list details:', error);
        toast({
          title: 'Error loading list',
          description: 'Please try again later',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        navigate('/lists');
      } finally {
        setIsLoading(false);
      }
    };

    fetchListDetails();
  }, [id, navigate, toast]);

  const handleAddWord = async (word: string): Promise<void> => {
    try {
      const newWord = await apiService.addWord(id!, word);
      setWords(prevWords => [newWord, ...prevWords]);
      toast({
        title: 'Word added successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      onClose();
    } catch (error) {
      console.error('Error adding word:', error);
      toast({
        title: 'Error adding word',
        description: 'Please try again later',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleDeleteWord = async (wordId: string) => {
    try {
      await apiService.deleteWord(id!, wordId);
      setWords(prevWords => prevWords.filter(word => word.id !== wordId));
      toast({
        title: 'Word deleted successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error deleting word:', error);
      toast({
        title: 'Error deleting word',
        description: 'Please try again later',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleDeleteList = async () => {
    if (!id || !window.confirm('Are you sure you want to delete this list? This action cannot be undone.')) return;
    
    try {
      await apiService.deleteList(id);
      toast({
        title: 'List deleted successfully',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      navigate('/lists');
    } catch (error) {
      console.error('Error deleting list:', error);
      toast({
        title: 'Error deleting list',
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

  if (!list) {
    return (
      <Container maxW="container.xl" py={8} px={{ base: 4, md: 8 }}>
        <Box textAlign="center" py={10}>
          <Text>List not found</Text>
          <Button 
            onClick={() => navigate('/lists')} 
            mt={4}
            colorScheme="green"
          >
            Back to Lists
          </Button>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxW="container.xl" py={8} px={{ base: 4, md: 8 }}>
      <MotionBox
        initial="hidden"
        animate="visible"
        variants={fadeIn}
        transition={{ duration: 0.5 }}
        p={4}
      >
        <Flex mb={6} justify="space-between" align="center">
          <Flex align="center" gap={2}>
            <IconButton
              aria-label="Go back"
              icon={<ArrowBackIcon />}
              variant="ghost"
              onClick={() => navigate(-1)}
              size="lg"
              _hover={{
                transform: 'translateY(-2px)',
                color: 'green.400'
              }}
              transition="all 0.2s"
            />
            <Heading 
              as="h1" 
              size="xl"
              bgGradient="linear(to-r, green.400, brand.400)"
              bgClip="text"
              display="flex"
              alignItems="center"
              gap={2}
            >
              <Icon 
                as={GiTreeBranch} 
                color="green.400"
                animation={`${sparkle} 3s ease infinite`}
              />
              {list.name}
            </Heading>
          </Flex>
          <IconButton
            aria-label="Delete list"
            icon={<DeleteIcon />}
            variant="ghost"
            colorScheme="red"
            onClick={handleDeleteList}
            size="lg"
            _hover={{
              bg: 'red.900',
              transform: 'scale(1.1)'
            }}
            transition="all 0.2s"
          />
        </Flex>

        <Flex 
          justify="space-between" 
          align="center" 
          mb={6}
          direction={{ base: 'column', md: 'row' }}
          gap={4}
        >
          <Box maxW="container.md">
            <Text color="gray.400" fontSize="lg">{list.description}</Text>
            {list.context && (
              <Text color="gray.500" fontSize="md" mt={2}>
                Context: {list.context}
              </Text>
            )}
          </Box>
          <Flex gap={3} flexWrap="wrap" justify={{ base: 'center', md: 'flex-end' }}>
            <Button 
              variant="ghost" 
              leftIcon={<FaGraduationCap />}
              colorScheme="green"
              _hover={{ 
                transform: 'translateY(-2px)',
              }}
              transition="all 0.2s"
              size="lg"
              isDisabled={words.length === 0}
              onClick={() => navigate(`/learn/${list!.id}`, { state: { list } })}
            >
              Learn
            </Button>
            <Button 
              variant="ghost"
              leftIcon={<FaGamepad />}
              colorScheme="orange"
              _hover={{ 
                transform: 'translateY(-2px)',
              }}
              transition="all 0.2s"
              size="lg"
              isDisabled={words.length === 0}
              onClick={() => navigate(`/quiz/${list!.id}`, { state: { list } })}
            >
              Quiz
            </Button>
            <Button 
              variant="solid"
              colorScheme="green"
              leftIcon={<FaPlus />}
              _hover={{ 
                transform: 'translateY(-2px)',
              }}
              transition="all 0.2s"
              size="lg"
              onClick={onOpen}
            >
              Add Word
            </Button>
          </Flex>
        </Flex>

        <Box 
          bg="slate.800"
          borderRadius="xl"
          borderWidth="1px"
          borderColor="slate.700"
          overflow="hidden"
          _hover={{ borderColor: 'slate.600' }}
          transition="all 0.2s"
        >
          {words.length === 0 ? (
            <Flex 
              direction="column" 
              align="center" 
              gap={4} 
              py={12}
              px={4}
            >
              <Icon 
                as={GiTreeBranch} 
                boxSize={12} 
                color="green.400" 
                animation={`${sparkle} 3s ease infinite`}
              />
              <Text color="gray.400" fontSize="lg" textAlign="center">
                Your tree is empty! Add some words to help it grow. 🌱
              </Text>
              <Button
                variant="outline"
                colorScheme="green"
                leftIcon={<FaPlus />}
                onClick={onOpen}
                size="lg"
                _hover={{
                  transform: 'translateY(-2px)',
                }}
                transition="all 0.2s"
              >
                Add Your First Word
              </Button>
            </Flex>
          ) : (
            <Box p={4}>
              {words.map((word: Word, index: number) => (
                <MotionBox
                  key={word.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  whileHover={{ 
                    x: 10,
                    backgroundColor: generateHoverColor(word.value),
                  }}
                  transition={{ 
                    duration: 0.3, 
                    delay: index * 0.1,
                    type: "tween",
                    ease: "easeOut"
                  }}
                  onClick={() => setSelectedWord(selectedWord === word.id ? null : word.id)}
                  p={4}
                  mb={2}
                  borderRadius="lg"
                  bg={generateColor(word.value)}
                  position="relative"
                  cursor="pointer"
                >
                  <Flex justify="space-between" align="center">
                    <Box w="full">
                      <Text 
                        fontSize="xl" 
                        fontWeight="bold" 
                        color="white"
                        mb={selectedWord === word.id ? 2 : 0}
                      >
                        {word.value}
                      </Text>
                      {selectedWord === word.id && (
                        <Text 
                          color="gray.200" 
                          fontSize="md"
                          transition="all 0.3s"
                        >
                          {word.meaning}
                        </Text>
                      )}
                    </Box>
                    <Box
                      onMouseEnter={(e) => {
                        e.stopPropagation();
                        setSelectedWord(word.id);
                      }}
                      onMouseLeave={(e) => {
                        e.stopPropagation();
                        if (selectedWord !== word.id) {
                          setSelectedWord(null);
                        }
                      }}
                    >
                      <IconButton
                        aria-label="Delete word"
                        icon={<DeleteIcon />}
                        variant="ghost"
                        colorScheme="red"
                        opacity={selectedWord === word.id ? 1 : 0}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteWord(word.id);
                        }}
                        _hover={{
                          transform: 'scale(1.1)',
                        }}
                        transition="all 0.2s"
                      />
                    </Box>
                  </Flex>
                </MotionBox>
              ))}
            </Box>
          )}
        </Box>

        <AddWordModal
          isOpen={isOpen}
          onClose={onClose}
          onAddWord={handleAddWord}
          listName={list?.name || ''}
        />
      </MotionBox>
    </Container>
  );
}; 
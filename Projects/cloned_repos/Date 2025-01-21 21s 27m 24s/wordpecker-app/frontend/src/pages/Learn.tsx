import { 
  Box, 
  Button, 
  Text, 
  Flex, 
  Progress, 
  Radio, 
  RadioGroup, 
  Stack, 
  Badge, 
  keyframes, 
  IconButton,
  useToast,
  Spinner,
  Center
} from '@chakra-ui/react';
import { useState, useEffect, useRef } from 'react';
import { useParams, Link, useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Exercise, WordList } from '../types';
import { ArrowBackIcon, CloseIcon } from '@chakra-ui/icons';
import { apiService } from '../services/api';

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const pulse = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const MotionBox = motion(Box);
const MotionFlex = motion(Flex);

export const Learn = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const { state } = useLocation();
  const toast = useToast();
  const hasInitializedRef = useRef(false);
  const isMountedRef = useRef(false);
  
  const [list] = useState<WordList | null>(state?.list || null);
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [currentExercise, setCurrentExercise] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState('');
  const [isAnswered, setIsAnswered] = useState(false);
  const [streak, setStreak] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [isCompleted, setIsCompleted] = useState(false);

  useEffect(() => {
    if (isMountedRef.current) return;
    isMountedRef.current = true;

    const initLearn = async () => {
      if (!id || !list || hasInitializedRef.current) return;
      
      try {
        setIsLoading(true);
        
        // Start learning session
        const response = await apiService.startLearning(id);
        if (response && response.exercises) {
          setExercises(response.exercises);
          hasInitializedRef.current = true;
        } else {
          throw new Error('Invalid response from server');
        }
      } catch (error: any) {
        console.error('Error initializing learning session:', error);
        toast({
          title: 'Error',
          description: error.response?.data?.message || 'Failed to start learning session',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        navigate(`/lists/${id}`);
      } finally {
        setIsLoading(false);
      }
    };

    initLearn();
  }, [id, navigate, toast]);

  const loadMoreExercises = async () => {
    if (!id || !exercises.length) return false;
    
    try {
      setIsLoading(true);
      const response = await apiService.getExercises(id);
      
      if (response && response.exercises && response.exercises.length > 0) {
        setExercises(prev => [...prev, ...response.exercises]);
        setCurrentExercise(exercises.length); // Start from the first new exercise
        setSelectedAnswer('');
        setIsAnswered(false);
        setIsCompleted(false);
        return true;
      }
      return false;
    } catch (error: any) {
      console.error('Error loading more exercises:', error);
      toast({
        title: 'Error',
        description: error.response?.data?.message || 'Failed to load more exercises',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnswer = () => {
    setIsAnswered(true);
    const exercise = exercises[currentExercise];
    if (selectedAnswer === exercise.correct_answer) {
      setStreak(prev => prev + 1);
    } else {
      setStreak(0);
    }
  };

  const handleNext = async () => {
    const isLastExercise = currentExercise === exercises.length - 1;
    
    if (isLastExercise) {
      setIsCompleted(true);
      return;
    }
    
    setCurrentExercise(prev => prev + 1);
    setSelectedAnswer('');
    setIsAnswered(false);
  };

  if (isLoading) {
    return (
      <Center h="calc(100vh - 64px)">
        <Spinner size="xl" color="green.400" thickness="4px" />
      </Center>
    );
  }

  if (!list || !exercises.length) {
    return (
      <Box textAlign="center" py={10}>
        <Text mb={4}>No exercises available</Text>
        <Link to="/lists">
          <Button variant="solid">Back to Lists</Button>
        </Link>
      </Box>
    );
  }

  const exercise = exercises[currentExercise];
  const progress = ((currentExercise + 1) / exercises.length) * 100;

  return (
    <MotionBox
      initial="hidden"
      animate="visible"
      variants={fadeIn}
      transition={{ duration: 0.5 }}
      p={4}
    >
      <Flex mb={4}>
        <IconButton
          aria-label="Go back"
          icon={<ArrowBackIcon />}
          variant="ghost"
          onClick={() => navigate(-1)}
          size="lg"
        />
      </Flex>

      <Flex 
        justify="space-between" 
        align="center" 
        mb={6}
        direction={{ base: 'column', md: 'row' }}
        gap={4}
      >
        <Box>
          <Text 
            textStyle="h1"
            color="white"
            fontSize={{ base: '3xl', md: '4xl' }}
          >
            Learning: {list.name}
          </Text>
          <Flex 
            gap={4} 
            mt={2} 
            flexWrap="wrap" 
            justify={{ base: 'center', md: 'flex-start' }}
          >
            <Badge 
              colorScheme="green" 
              p={2} 
              borderRadius="full"
              animation={streak > 0 ? `${pulse} 1s ease infinite` : undefined}
            >
              🔥 Streak: {streak}
            </Badge>
            <Badge colorScheme="blue" p={2} borderRadius="full">
              ✨ Progress: {Math.round(progress)}%
            </Badge>
          </Flex>
        </Box>
        <Link to={`/lists/${id}`}>
          <IconButton
            aria-label="Exit"
            icon={<CloseIcon />}
            variant="ghost"
            size="lg"
          />
        </Link>
      </Flex>

      <Progress 
        value={progress} 
        mb={8} 
        rounded="full" 
        size="sm"
        colorScheme="green"
        hasStripe
        isAnimated
      />

      <MotionBox
        layerStyle="card"
        maxW="800px"
        mx="auto"
        variants={fadeIn}
        initial="hidden"
        animate="visible"
        boxShadow="2xl"
        borderWidth="1px"
        borderColor="green.800"
        px={{ base: 4, md: 8 }}
        py={6}
      >
        <Text 
          fontSize={{ base: 'xl', md: '2xl' }}
          mb={6}
          textAlign="center"
          fontWeight="bold"
          bgGradient="linear(to-r, green.200, teal.200)"
          bgClip="text"
        >
          {exercise.question}
        </Text>

        <RadioGroup 
          value={selectedAnswer} 
          onChange={setSelectedAnswer} 
          isDisabled={isAnswered}
        >
          <Stack spacing={4}>
            {exercise.options?.map((option) => (
              <MotionFlex
                key={option}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                w="100%"
              >
                <Radio
                  value={option}
                  size="lg"
                  w="100%"
                  p={4}
                  borderWidth={2}
                  borderRadius="lg"
                  borderColor={
                    isAnswered
                      ? option === exercise.correct_answer
                        ? "green.500"
                        : selectedAnswer === option
                        ? "red.500"
                        : "transparent"
                      : "transparent"
                  }
                  bg={
                    isAnswered
                      ? option === exercise.correct_answer
                        ? "green.900"
                        : selectedAnswer === option
                        ? "red.900"
                        : "slate.700"
                      : "slate.700"
                  }
                  _hover={{
                    bg: isAnswered ? undefined : "slate.600"
                  }}
                >
                  <Text ml={2} fontSize={{ base: 'md', md: 'lg' }}>
                    {option}
                  </Text>
                </Radio>
              </MotionFlex>
            ))}
          </Stack>
        </RadioGroup>

        {isAnswered && (
          <MotionBox
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            mt={6}
            p={4}
            bg={selectedAnswer === exercise.correct_answer ? 'green.900' : 'red.900'}
            borderRadius="lg"
            textAlign="center"
          >
            <Text color="white" fontSize={{ base: 'lg', md: 'xl' }} fontWeight="bold">
              {selectedAnswer === exercise.correct_answer 
                ? '🎉 Excellent!' 
                : '💡 Keep Learning!'}
            </Text>
            <Text color="white" mt={2} fontSize={{ base: 'sm', md: 'md' }}>
              {selectedAnswer === exercise.correct_answer 
                ? streak > 1 ? `You're on fire! ${streak} correct in a row!` : 'Great job!'
                : `The correct answer is: ${exercise.correct_answer}`}
            </Text>
          </MotionBox>
        )}

        <Flex justify="center" mt={8} gap={4}>
          {!isAnswered ? (
            <Button
              variant="solid"
              colorScheme="blue"
              size="lg"
              onClick={handleAnswer}
              isDisabled={!selectedAnswer}
              _hover={{
                transform: 'translateY(-2px)',
                shadow: 'lg'
              }}
              transition="all 0.2s"
            >
              Check Answer
            </Button>
          ) : isCompleted ? (
            <>
              <Button
                variant="outline"
                colorScheme="green"
                size="lg"
                onClick={() => navigate(`/lists/${id}`)}
                _hover={{
                  transform: 'translateY(-2px)',
                  shadow: 'lg'
                }}
                transition="all 0.2s"
              >
                Complete
              </Button>
              <Button
                variant="solid"
                colorScheme="blue"
                size="lg"
                onClick={loadMoreExercises}
                isLoading={isLoading}
                _hover={{
                  transform: 'translateY(-2px)',
                  shadow: 'lg'
                }}
                transition="all 0.2s"
              >
                Continue Learning
              </Button>
            </>
          ) : (
            <Button
              variant="solid"
              colorScheme="green"
              size="lg"
              onClick={handleNext}
              _hover={{
                transform: 'translateY(-2px)',
                shadow: 'lg'
              }}
              transition="all 0.2s"
            >
              Next Exercise
            </Button>
          )}
        </Flex>
      </MotionBox>
    </MotionBox>
  );
}; 
import { 
  Box, 
  Button, 
  Text, 
  Flex, 
  Progress, 
  Radio, 
  RadioGroup, 
  Stack, 
  HStack, 
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
import { Question, WordList } from '../types';
import { ArrowBackIcon, CloseIcon } from '@chakra-ui/icons';
import { apiService } from '../services/api';

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const bounce = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
`;

const sparkle = keyframes`
  0% { transform: scale(1) rotate(0deg); }
  25% { transform: scale(1.2) rotate(-5deg); }
  50% { transform: scale(1) rotate(0deg); }
  75% { transform: scale(1.2) rotate(5deg); }
  100% { transform: scale(1) rotate(0deg); }
`;

const MotionBox = motion(Box);
const MotionFlex = motion(Flex);

export const Quiz = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const { state } = useLocation();
  const toast = useToast();
  const hasInitializedRef = useRef(false);
  const isMountedRef = useRef(false);
  
  const [list] = useState<WordList | null>(state?.list || null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState('');
  const [isAnswered, setIsAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [lives, setLives] = useState(3);
  const [combo, setCombo] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [totalQuestions, setTotalQuestions] = useState(0);

  useEffect(() => {
    if (isMountedRef.current) return;
    isMountedRef.current = true;

    const initQuiz = async () => {
      if (!id || !list || hasInitializedRef.current) return;
      
      try {
        setIsLoading(true);
        
        // Start quiz session
        const response = await apiService.startQuiz(id);
        if (response && response.questions && response.total_questions) {
          setQuestions(response.questions);
          setTotalQuestions(response.total_questions);
          hasInitializedRef.current = true;
        } else {
          throw new Error('Invalid response from server');
        }
      } catch (error: any) {
        console.error('Error initializing quiz:', error);
        toast({
          title: 'Error',
          description: error.response?.data?.message || 'Failed to start quiz',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        navigate(`/lists/${id}`);
      } finally {
        setIsLoading(false);
      }
    };

    initQuiz();
  }, [id, navigate, toast]);

  const loadMoreQuestions = async () => {
    if (!id || !questions.length) return false;
    
    try {
      const response = await apiService.getQuestions(id);
      
      if (response && response.questions && response.questions.length > 0) {
        setQuestions(prev => [...prev, ...response.questions]);
        return !response.completed;
      }
      return false;
    } catch (error: any) {
      console.error('Error loading more questions:', error);
      toast({
        title: 'Error',
        description: error.response?.data?.message || 'Failed to load more questions',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return false;
    }
  };

  const handleAnswer = () => {
    setIsAnswered(true);
    const question = questions[currentQuestion];
    if (selectedAnswer === question.correct_answer) {
      setScore(prev => prev + (combo > 2 ? 150 : combo > 1 ? 120 : 100));
      setCombo(prev => prev + 1);
    } else {
      setLives(prev => prev - 1);
      setCombo(0);
    }
  };

  const handleNext = async () => {
    if (lives === 0) {
      navigate(`/lists/${id}`);
      return;
    }
    
    const isLastQuestion = currentQuestion === questions.length - 1;
    if (isLastQuestion && currentQuestion + 1 < totalQuestions) {
      // Load more questions before proceeding
      const hasMoreQuestions = await loadMoreQuestions();
      if (!hasMoreQuestions && currentQuestion + 1 >= questions.length) {
        navigate(`/lists/${id}`);
        return;
      }
    }
    
    if (currentQuestion + 1 >= totalQuestions) {
      navigate(`/lists/${id}`);
    } else {
      setCurrentQuestion(prev => prev + 1);
      setSelectedAnswer('');
      setIsAnswered(false);
    }
  };

  if (isLoading) {
    return (
      <Center h="calc(100vh - 64px)">
        <Spinner size="xl" color="purple.400" thickness="4px" />
      </Center>
    );
  }

  if (!list || !questions.length) {
    return (
      <Box textAlign="center" py={10}>
        <Text mb={4}>No questions available</Text>
        <Link to="/lists">
          <Button variant="solid">Back to Lists</Button>
        </Link>
      </Box>
    );
  }

  const question = questions[currentQuestion];
  const progress = ((currentQuestion + 1) / totalQuestions) * 100;

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
            Quiz: {list.name}
          </Text>
          <HStack spacing={4} mt={2} flexWrap="wrap" justify={{ base: 'center', md: 'flex-start' }}>
            <Badge 
              colorScheme="purple" 
              p={2} 
              borderRadius="full"
              animation={combo > 2 ? `${sparkle} 1s ease infinite` : undefined}
            >
              ⚡ Combo x{combo}
            </Badge>
            <Badge 
              colorScheme="yellow" 
              p={2} 
              borderRadius="full"
              animation={score > 0 ? `${bounce} 1s ease infinite` : undefined}
            >
              🏆 Score: {score}
            </Badge>
            <Badge 
              colorScheme="red" 
              p={2} 
              borderRadius="full"
            >
              {"❤️".repeat(lives)}
            </Badge>
          </HStack>
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
        colorScheme="purple"
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
        borderColor="purple.800"
        px={{ base: 4, md: 8 }}
        py={6}
      >
        <Text 
          fontSize={{ base: 'xl', md: '2xl' }}
          mb={6}
          textAlign="center"
          fontWeight="bold"
          color="white"
        >
          {question.question}
        </Text>

        <RadioGroup 
          value={selectedAnswer} 
          onChange={setSelectedAnswer} 
          isDisabled={isAnswered}
        >
          <Stack spacing={4}>
            {question.options?.map((option) => (
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
                      ? option === question.correct_answer
                        ? "purple.500"
                        : selectedAnswer === option
                        ? "red.500"
                        : "transparent"
                      : "transparent"
                  }
                  bg={
                    isAnswered
                      ? option === question.correct_answer
                        ? "purple.900"
                        : selectedAnswer === option
                        ? "red.900"
                        : "slate.700"
                      : "slate.700"
                  }
                  _hover={{
                    bg: isAnswered ? undefined : "slate.600",
                    transform: "translateX(8px)"
                  }}
                  transition="all 0.2s"
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
            bg={selectedAnswer === question.correct_answer ? 'purple.900' : 'red.900'}
            borderRadius="lg"
            textAlign="center"
          >
            <Text color="white" fontSize={{ base: 'lg', md: 'xl' }} fontWeight="bold">
              {selectedAnswer === question.correct_answer 
                ? `🎉 Correct! ${combo > 2 ? '⚡ Perfect combo!' : combo > 1 ? '⚡ Great combo!' : ''}`
                : '❌ Incorrect. Try again!'}
            </Text>
          </MotionBox>
        )}

        <Flex justify="center" mt={8}>
          <Button
            colorScheme={isAnswered ? 'purple' : 'blue'}
            size="lg"
            onClick={isAnswered ? handleNext : handleAnswer}
            isDisabled={!selectedAnswer}
            _hover={{
              transform: 'translateY(-2px)',
              shadow: 'lg'
            }}
          >
            {isAnswered 
              ? lives === 0 
                ? 'Game Over' 
                : currentQuestion + 1 >= totalQuestions 
                  ? 'Finish Quiz' 
                  : 'Next Question'
              : 'Check Answer'}
          </Button>
        </Flex>
      </MotionBox>
    </MotionBox>
  );
}; 
import { Router } from 'express';
import { validate } from "../middleware/validate";
import { quizController } from '../controllers/quizController';

const router = Router();

// Quiz routes
router.post('/:listId/start', validate.listId, quizController.startQuiz);
router.post('/:listId/more', validate.listId, quizController.getMoreQuestions);

export default router; 
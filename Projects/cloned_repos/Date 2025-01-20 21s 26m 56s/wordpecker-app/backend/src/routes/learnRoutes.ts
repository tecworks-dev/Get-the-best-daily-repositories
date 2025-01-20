import { Router } from 'express';
import { validate } from "../middleware/validate";
import { learnController } from '../controllers/learnController';

const router = Router();

// Learning routes
router.post('/:listId/start', validate.listId, learnController.startLearning);
router.post('/:listId/more', validate.listId, learnController.getMoreExercises);

export default router; 
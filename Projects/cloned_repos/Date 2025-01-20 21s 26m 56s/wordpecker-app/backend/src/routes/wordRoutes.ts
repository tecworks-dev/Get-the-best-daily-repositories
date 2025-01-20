import { Router } from 'express';
import { validate } from '../middleware/validate';
import { wordController } from '../controllers/wordController';

const router = Router();

// Word routes
router.post('/:listId/words', 
  validate.listId,
  validate.addWord,
  wordController.addWord
);

router.get('/:listId/words', 
  validate.listId,
  wordController.getWords
);

router.delete('/:listId/words/:wordId', 
  validate.listId,
  validate.wordId,
  wordController.deleteWord
);

export default router; 
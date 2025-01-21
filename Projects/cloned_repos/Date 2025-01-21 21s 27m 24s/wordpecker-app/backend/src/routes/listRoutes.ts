import { Router } from 'express';
import { validate } from "../middleware/validate";
import { listController } from '../controllers/listController';

const router = Router();

// List routes
router.post('/', validate.createList, listController.createList);
router.get('/', listController.getLists);
router.get('/:id', validate.id, listController.getList);
router.put('/:id', validate.id, validate.updateList, listController.updateList);
router.delete('/:id', validate.id, listController.deleteList);

export default router; 
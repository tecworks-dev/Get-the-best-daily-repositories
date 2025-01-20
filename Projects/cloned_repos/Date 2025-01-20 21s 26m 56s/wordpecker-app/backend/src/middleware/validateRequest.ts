import { Request, Response, NextFunction } from 'express';
import { validationResult, ValidationError } from 'express-validator';

export const validateRequest = (req: Request, res: Response, next: NextFunction) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    const firstError = errors.array()[0] as ValidationError;
    
    // If it's a UUID validation error for either id or listId parameter, return 404
    if (firstError.type === 'field' && 
        (firstError.path === 'id' || firstError.path === 'listId') && 
        firstError.msg.includes('UUID')) {
      return res.status(404).json({ message: 'List not found' });
    }
    
    // For other validation errors, return 400
    return res.status(400).json({ errors: errors.array() });
  }
  next();
}; 
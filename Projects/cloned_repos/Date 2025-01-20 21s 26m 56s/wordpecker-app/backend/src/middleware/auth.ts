import { Request, Response, NextFunction } from 'express';
import { supabase } from '../config/supabase';

export const authMiddleware = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader?.startsWith('Bearer ')) {
      return res.status(401).json({ message: 'No token provided' });
    }

    const token = authHeader.split(' ')[1];

    // For tests, allow test token
    if (process.env.NODE_ENV === 'test' && token === process.env.TEST_USER_TOKEN) {
      req.user = { id: process.env.TEST_USER_UUID! };
      return next();
    }

    // Verify token with Supabase
    const { data: { user }, error } = await supabase.auth.getUser(token);

    if (error || !user) {
      return res.status(401).json({ message: 'Invalid token' });
    }

    req.user = { id: user.id };
    next();
  } catch (error) {
    console.error('Auth error:', error);
    res.status(500).json({ message: 'Error authenticating request' });
  }
}; 
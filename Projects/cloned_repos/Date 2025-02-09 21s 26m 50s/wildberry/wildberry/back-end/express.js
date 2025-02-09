import express, { Request, Response } from 'express';
import bodyParser from 'body-parser';

const app = express();
const port = 3000; // Or any other port you prefer

app.use(bodyParser.json());

// --- Data Models (Replace with your actual data models and persistence layer) ---

interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  currencyCode: string;
  // ... other product properties ...
}

interface Customer {
  id: string;
  email?: string | null;
  displayName?: string | null;
}

interface PurchaseResult {
  productId: string;
  customerInfo: Customer; // Simplified
}

// --- In-Memory Data Store (Replace with Database Interaction) ---

const products: Product[] = [
  { id: 'product_1', name: 'Product 1', description: 'Description 1', price: 9.99, currencyCode: 'USD' },
  { id: 'product_2', name: 'Product 2', description: 'Description 2', price: 19.99, currencyCode: 'USD' },
  { id: 'annual_sub', name: 'Annual Subscription', description: 'Annual sub description', price: 99.99, currencyCode: 'USD' },
  { id: 'monthly_sub', name: 'Monthly Subscription', description: 'Monthly sub description', price: 9.99, currencyCode: 'USD' },
];

// --- Helper Functions ---

const getProductById = (productId: string): Product | undefined => {
  return products.find((p) => p.id === productId);
};

// --- API Endpoints ---

// Get Offerings
app.get('/offerings', (req: Request, res: Response) => {
  const offerings = {
    all: {},
    current: {}
  };

  res.json(offerings);
});

// Get Products
app.get('/products', (req: Request, res: Response) => {
  const productIds: string[] = req.query.productIds as string[] || [];

  if (!Array.isArray(productIds)) {
    return res.status(400).json({ error: 'productIds must be an array' });
  }

  const requestedProducts = products.filter((product) =>
    productIds.includes(product.id)
  );

  res.json(requestedProducts);
});

// Purchase Product  (Simplified - no payment processing, no receipt validation)
app.post('/purchase', (req: Request, res: Response) => {
  const { productId, appUserID } = req.body; // Expect these in the request body

  if (!productId || typeof productId !== 'string') {
    return res.status(400).json({ error: 'productId is required and must be a string' });
  }
  if (!appUserID || typeof appUserID !== 'string') {
    return res.status(400).json({ error: 'appUserID is required and must be a string' });
  }

  const product = getProductById(productId);
  if (!product) {
    return res.status(404).json({ error: 'Product not found' });
  }

  const purchaseResult: PurchaseResult = {
    productId: product.id,
    customerInfo: { id: appUserID, email: 'test@example.com' }, // VERY simplified customer info
  }

  res.status(201).json(purchaseResult); // 201 Created
});

// Restore Purchases (Simplified - no actual restore logic)
app.post('/restore', (req: Request, res: Response) => {
  const { appUserID } = req.body;

  if (!appUserID) {
    return res.status(400).send({ message: 'appUserID is required' });
  }
  const customerInfo: Customer = {
    id: appUserID,
  };
  res.json(customerInfo); // Return *relevant* restored purchase data.
});

// Get Customer Info (Simplified)
app.get('/customer/:appUserID', (req: Request, res: Response) => {
  const { appUserID } = req.params;

  if (!appUserID) {
    return res.status(400).json({ error: 'appUserID is required' });
  }

  const customerInfo: Customer = {
    id: appUserID,
    displayName: 'Test User'
  };

  res.json(customerInfo);
});

// --- Error Handling ---

// Catch-all for 404s (Not Found)
app.use((req: Request, res: Response) => {
  res.status(404).json({ error: 'Not Found' });
});

// General error handler
app.use((err: Error, req: Request, res: Response, next: Function) => {
  console.error(err.stack); // Log the error for debugging
  res.status(500).json({ error: 'Internal Server Error' });
});

// --- Start the Server ---

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

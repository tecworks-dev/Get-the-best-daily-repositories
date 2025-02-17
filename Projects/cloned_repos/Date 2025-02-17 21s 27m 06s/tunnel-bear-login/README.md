# Tunnel Bear Login Animation

A delightful login form implementation inspired by [The Tunnel Bear](https://www.tunnelbear.com/account/login) by Kadri Jibraan. This project recreates the charming bear animation that responds to user input and focus states.

## Features

- Interactive bear animation that responds to email input length
- Playful hide animation when focusing on the password field
- Clean, modern UI with Tailwind CSS

## Project Structure

The project follows React best practices with a modular component architecture:

```
src/
├── components/
│   ├── BearAvatar.tsx    # Animated bear image component
│   ├── Input.tsx         # Reusable form input component
│   └── LoginForm.tsx     # Main login form component
├── hooks/
│   ├── useBearAnimation.ts   # Bear animation state management
│   └── useBearImages.ts      # Image loading and sorting logic
└── assets/
    └── img/              # Bear animation image sequences
```

## Technical Details

- Built with React 19 and TypeScript
- Styled using Tailwind CSS
- Vite for fast development and building
- Modular architecture with custom hooks for state management
- Optimized image loading and sorting

## Development

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

## Credits

This project is a React implementation inspired by [The Tunnel Bear](https://www.tunnelbear.com/account/login) login form created by Kadri Jibraan. All bear animations and design concepts are credited to the original work.

## License

This project is for educational purposes only. The original design and animations are property of TunnelBear.

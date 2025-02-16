import { forwardRef, InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  error?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { error, className = '', ...props },
  ref
) {
  return (
    <div className="w-full">
      <input
        ref={ref}
        className={`w-full px-4 py-2 rounded border border-gray-300 focus:border-tunnel-bear focus:ring-1 focus:ring-tunnel-bear focus:outline-none ${className}`}
        {...props}
      />
      {error && (
        <p className="mt-1 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
});

export default Input;

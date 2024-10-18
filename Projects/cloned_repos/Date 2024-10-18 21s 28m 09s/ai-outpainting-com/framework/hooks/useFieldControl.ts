import {
  Control,
  FieldPath,
  FieldValues,
  RegisterOptions,
  useController,
} from 'react-hook-form'

/**
 * Custom hook to work with controlled component,
 * this function provide you with both form and field level state.
 * Re-render is isolated at the hook level.
 * @param name
 * @param control
 * @param rules
 */
export default function useFieldControl<
  TFieldValues extends FieldValues = FieldValues,
  TContext = any,
  TTransformedValues extends FieldValues | undefined = undefined,
>(
  name: FieldPath<TFieldValues>,
  control: Control<TFieldValues, TContext>,
  rules?: Omit<
    RegisterOptions<TFieldValues, FieldPath<TFieldValues>>,
    'valueAsNumber' | 'valueAsDate' | 'setValueAs' | 'disabled'
  >,
) {
  const { field } = useController({ name, control, rules })
  return field
}

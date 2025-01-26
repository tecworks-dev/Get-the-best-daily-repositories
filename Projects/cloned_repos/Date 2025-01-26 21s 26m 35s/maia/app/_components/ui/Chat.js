export const Chat = ({ result }) => {
  return (
    <div className="flex items-end gap-2">
      <div className="break-words rounded-2xl bg-transparent text-white max-w-full bg-background-50 whitespace-pre-wrap w-full">
        {result}
      </div>
    </div>
  );
};

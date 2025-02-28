export const DemoWidget = () => {
  return (
    <div className="bg-white rounded-xl shadow-xl p-6">
      <h3 className="text-xl font-semibold mb-4">Try It Out</h3>
      <div className="space-y-4">
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Sample Feedback
            </label>
            <textarea 
              className="w-full p-3 border border-slate-200 rounded-lg"
              rows={3}
              placeholder="Paste user feedback here..."
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Generated Code
            </label>
            <pre className="p-3 bg-slate-900 text-slate-100 rounded-lg text-sm">
              <code>{/* Code will appear here... */}</code>
            </pre>
          </div>
        </div>
        <button className="w-full py-2 bg-primary hover:bg-primary-dark text-white rounded-lg transition-colors">
          Generate Code
        </button>
      </div>
    </div>
  );
}; 
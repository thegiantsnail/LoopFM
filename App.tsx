import React from 'react';
import LoopVisualizer from './components/LoopVisualizer';

const App: React.FC = () => {
  return (
    <div className="w-full h-screen bg-[#0b0f14] text-slate-200 overflow-hidden font-sans">
      <LoopVisualizer />
    </div>
  );
};

export default App;
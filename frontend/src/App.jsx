import React from 'react';

function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 flex flex-col items-center justify-center space-y-12">

      {/* Flexbox row */}
      <div className="flex flex-row gap-6">
        <div className="flex-1 bg-indigo-500 p-6 rounded-lg text-center hover:bg-indigo-600 transition">
          Flex Item 1
        </div>
        <div className="flex-1 bg-pink-500 p-6 rounded-lg text-center hover:bg-pink-600 transition">
          Flex Item 2
        </div>
        <div className="flex-1 bg-green-500 p-6 rounded-lg text-center hover:bg-green-600 transition">
          Flex Item 3
        </div>
      </div>

      {/* Grid example */}
      <div className="grid grid-cols-3 gap-6 w-full max-w-4xl">
        <div className="bg-yellow-500 p-6 rounded-lg text-center hover:bg-yellow-600 transition">
          Grid Item 1
        </div>
        <div className="bg-purple-500 p-6 rounded-lg text-center hover:bg-purple-600 transition">
          Grid Item 2
        </div>
        <div className="bg-teal-500 p-6 rounded-lg text-center hover:bg-teal-600 transition">
          Grid Item 3
        </div>
        <div className="bg-orange-500 p-6 rounded-lg text-center hover:bg-orange-600 transition">
          Grid Item 4
        </div>
        <div className="bg-red-500 p-6 rounded-lg text-center hover:bg-red-600 transition">
          Grid Item 5
        </div>
        <div className="bg-blue-500 p-6 rounded-lg text-center hover:bg-blue-600 transition">
          Grid Item 6
        </div>
      </div>

      {/* Nested flex + spacing */}
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-1 bg-gray-700 p-6 rounded-lg text-center hover:bg-gray-600 transition">
          Nested Flex 1
        </div>
        <div className="flex-1 bg-gray-500 p-6 rounded-lg text-center hover:bg-gray-400 transition">
          Nested Flex 2
        </div>
      </div>

    </div>
  );
}

export default App;

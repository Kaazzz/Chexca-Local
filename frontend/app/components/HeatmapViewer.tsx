'use client'

import { useState } from 'react'
import { Eye, Layers } from 'lucide-react'

interface HeatmapViewerProps {
  originalImage: string
  heatmapOverlay: string
  topDisease: string
}

export default function HeatmapViewer({ originalImage, heatmapOverlay, topDisease }: HeatmapViewerProps) {
  const [showOverlay, setShowOverlay] = useState(true)

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Eye className="w-6 h-6 text-primary-600" />
            <h3 className="text-2xl font-bold text-gray-800">
              Visual Explanation (Grad-CAM++)
            </h3>
          </div>

          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
              ${showOverlay
                ? 'bg-gradient-medical text-white shadow-lg'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }
            `}
          >
            <Layers className="w-5 h-5" />
            {showOverlay ? 'Hide Heatmap' : 'Show Heatmap'}
          </button>
        </div>

        <div className="relative bg-gray-900 rounded-xl overflow-hidden">
          <img
            src={showOverlay ? heatmapOverlay : originalImage}
            alt="X-ray Analysis"
            className="w-full h-auto max-h-[600px] object-contain mx-auto"
          />

          <div className="absolute bottom-4 left-4 bg-black/70 backdrop-blur-sm text-white px-4 py-2 rounded-lg">
            <p className="text-sm font-medium">Focus Area: {(topDisease || 'Unknown').replace('_', ' ')}</p>
          </div>
        </div>

        <div className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6">
          <h4 className="font-bold text-gray-800 mb-3 flex items-center gap-2">
            <span className="w-2 h-2 bg-gradient-medical rounded-full"></span>
            Understanding the Heatmap
          </h4>
          <p className="text-gray-700 leading-relaxed">
            The colored overlay (Grad-CAM++) highlights the regions of the X-ray that most influenced the AI's diagnosis.
            <span className="font-semibold text-red-600"> Warmer colors (red/yellow)</span> indicate areas of higher importance,
            while <span className="font-semibold text-blue-600">cooler colors (blue/green)</span> show less significant regions.
            This advanced visualization technique helps radiologists understand what the AI is "looking at" when making its predictions.
          </p>
        </div>
      </div>
    </div>
  )
}

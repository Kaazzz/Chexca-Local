'use client'

import { useState } from 'react'
import { Eye, Layers, Image as ImageIcon, Info } from 'lucide-react'

interface HeatmapViewerProps {
  originalImage: string
  heatmapOverlay: string
  topDisease: string
}

export default function HeatmapViewer({ originalImage, heatmapOverlay, topDisease }: HeatmapViewerProps) {
  const [showOverlay, setShowOverlay] = useState(true)
  const [imageLoaded, setImageLoaded] = useState(false)

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="bg-white rounded-2xl shadow-xl p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-medical rounded-lg">
              <Eye className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-800">
                Visual Explanation
              </h3>
              <p className="text-sm text-gray-600">Grad-CAM++ Activation Map</p>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setShowOverlay(true)}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
                ${showOverlay
                  ? 'bg-gradient-medical text-white shadow-lg'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }
              `}
            >
              <Layers className="w-5 h-5" />
              Heatmap
            </button>
            <button
              onClick={() => setShowOverlay(false)}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
                ${!showOverlay
                  ? 'bg-gradient-medical text-white shadow-lg'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }
              `}
            >
              <ImageIcon className="w-5 h-5" />
              Original
            </button>
          </div>
        </div>

        {/* Image Viewer */}
        <div className="relative bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl overflow-hidden shadow-2xl">
          {!imageLoaded && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="animate-pulse text-white">Loading visualization...</div>
            </div>
          )}
          
          <img
            src={showOverlay ? heatmapOverlay : originalImage}
            alt={showOverlay ? "Grad-CAM Heatmap Overlay" : "Original X-ray"}
            className="w-full h-auto max-h-[600px] object-contain mx-auto transition-opacity duration-300"
            onLoad={() => setImageLoaded(true)}
            style={{ opacity: imageLoaded ? 1 : 0 }}
          />

          {/* Info Badge */}
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm text-white px-4 py-2 rounded-lg shadow-lg">
            <p className="text-xs font-medium uppercase tracking-wide opacity-80">
              {showOverlay ? 'Activation Heatmap' : 'Original Image'}
            </p>
          </div>

          {/* Focus Area Badge */}
          {showOverlay && (
            <div className="absolute bottom-4 left-4 bg-gradient-medical backdrop-blur-sm text-white px-4 py-3 rounded-lg shadow-lg">
              <p className="text-xs uppercase tracking-wide opacity-90 mb-1">Primary Focus</p>
              <p className="text-sm font-bold">
                {(topDisease || 'Unknown').replace(/_/g, ' ')}
              </p>
            </div>
          )}

          {/* Heatmap Legend */}
          {showOverlay && (
            <div className="absolute bottom-4 right-4 bg-black/70 backdrop-blur-sm text-white px-4 py-3 rounded-lg shadow-lg">
              <p className="text-xs font-medium mb-2">Attention Intensity</p>
              <div className="flex items-center gap-2">
                <span className="text-xs">Low</span>
                <div className="w-24 h-3 rounded-full bg-gradient-to-r from-blue-500 via-yellow-500 to-red-500"></div>
                <span className="text-xs">High</span>
              </div>
            </div>
          )}
        </div>

        {/* Explanation Card */}
        <div className="mt-6 bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 rounded-xl p-6 border border-purple-100">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-white rounded-lg shadow-sm">
              <Info className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <h4 className="font-bold text-gray-800 mb-2">
                Understanding Grad-CAM++ Visualization
              </h4>
              <p className="text-gray-700 leading-relaxed text-sm">
                The colored heatmap overlay shows <span className="font-semibold">where the AI model focused</span> when analyzing this X-ray. 
                <span className="font-semibold text-red-600"> Red and yellow regions</span> indicate areas of highest attention that most influenced the diagnosis, 
                while <span className="font-semibold text-blue-600">blue and green areas</span> had less impact on the prediction.
              </p>
              <div className="mt-3 pt-3 border-t border-purple-200">
                <p className="text-xs text-gray-600">
                  💡 <span className="font-medium">Clinical Note:</span> This visualization helps radiologists verify that the AI is examining clinically relevant regions, 
                  enhancing trust and interpretability in AI-assisted diagnosis.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

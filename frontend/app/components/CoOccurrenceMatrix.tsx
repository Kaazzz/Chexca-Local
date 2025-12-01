'use client'

import { GitBranch } from 'lucide-react'

interface CoOccurrenceMatrixProps {
  coOccurrence: number[][]
  diseaseClasses: string[]
}

export default function CoOccurrenceMatrix({ coOccurrence, diseaseClasses }: CoOccurrenceMatrixProps) {
  const getColorIntensity = (value: number) => {
    const intensity = Math.min(value, 1)

    // More contrasting violet gradient: Light lavender to deep purple
    // Low values: Light lavender (#E0D4F7)
    // High values: Deep violet (#6B21A8)

    const r = Math.round(224 - (117 * intensity)) // 224 -> 107
    const g = Math.round(212 - (179 * intensity)) // 212 -> 33
    const b = Math.round(247 - (79 * intensity))  // 247 -> 168

    return `rgb(${r}, ${g}, ${b})`
  }

  // Check if we have valid co-occurrence data
  if (!coOccurrence || coOccurrence.length === 0 || diseaseClasses.length < 2) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="flex items-center gap-3 mb-6">
            <GitBranch className="w-6 h-6 text-primary-600" />
            <h3 className="text-2xl font-bold text-gray-800">
              Disease Co-Occurrence Analysis
            </h3>
          </div>
          <div className="text-center py-12">
            <p className="text-gray-500">
              Co-occurrence analysis requires multiple disease classes.
              <br />
              Your model outputs {diseaseClasses.length} class(es).
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <GitBranch className="w-6 h-6 text-primary-600" />
          <h3 className="text-2xl font-bold text-gray-800">
            Disease Co-Occurrence Analysis
          </h3>
        </div>

        <p className="text-gray-600 mb-6 leading-relaxed">
          This matrix shows the correlation between different diseases in the analysis.
          Darker cells indicate stronger co-occurrence patterns.
        </p>

        <div className="overflow-x-auto">
          <div className="min-w-max">
            {/* Header row */}
            <div className="flex mb-1">
              <div className="w-32"></div>
              {diseaseClasses.map((disease, idx) => (
                <div
                  key={idx}
                  className="w-16 h-32 flex items-end justify-center"
                >
                  <span className="text-xs text-gray-600 writing-mode-vertical transform rotate-180 whitespace-nowrap">
                    {disease.replace('_', ' ')}
                  </span>
                </div>
              ))}
            </div>

            {/* Matrix rows */}
            {diseaseClasses.map((rowDisease, rowIdx) => (
              <div key={rowIdx} className="flex items-center mb-1">
                <div className="w-32 text-xs text-gray-600 pr-2 text-right font-medium">
                  {rowDisease.replace('_', ' ')}
                </div>
                {diseaseClasses.map((colDisease, colIdx) => {
                  const value = coOccurrence[rowIdx]?.[colIdx] ?? 0
                  return (
                    <div
                      key={colIdx}
                      className="w-16 h-16 border border-gray-200 flex items-center justify-center group relative cursor-pointer transition-transform hover:scale-110 hover:z-10"
                      style={{
                        backgroundColor: getColorIntensity(value)
                      }}
                    >
                      <span className="text-xs font-semibold text-gray-700">
                        {(value * 100).toFixed(0)}
                      </span>

                      {/* Tooltip */}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-20">
                        <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 whitespace-nowrap shadow-xl">
                          <p className="font-semibold">{rowDisease.replace('_', ' ')}</p>
                          <p className="opacity-75">↔</p>
                          <p className="font-semibold">{colDisease.replace('_', ' ')}</p>
                          <p className="mt-1 text-center">
                            {(value * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="mt-8 flex flex-col items-center gap-3">
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-gray-700">Low Correlation</span>
            <div className="flex gap-1">
              {[0, 0.2, 0.4, 0.6, 0.8, 1].map((val, idx) => (
                <div
                  key={idx}
                  className="w-16 h-8 border-2 border-gray-300 rounded shadow-sm"
                  style={{ backgroundColor: getColorIntensity(val) }}
                >
                  <div className="text-[10px] text-center pt-1 font-semibold text-gray-700">
                    {Math.round(val * 100)}%
                  </div>
                </div>
              ))}
            </div>
            <span className="text-sm font-medium text-gray-700">High Correlation</span>
          </div>
          <p className="text-xs text-gray-500 max-w-2xl text-center">
            Darker shades indicate stronger co-occurrence patterns between diseases
          </p>
        </div>
      </div>
    </div>
  )
}

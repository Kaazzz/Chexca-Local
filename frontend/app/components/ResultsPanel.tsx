'use client'

import { AnalysisResult } from '../lib/api'
import { formatPercentage, getConfidenceColor, getConfidenceBgColor } from '../lib/utils'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react'

interface ResultsPanelProps {
  result: AnalysisResult
}

export default function ResultsPanel({ result }: ResultsPanelProps) {
  // Sanitize chart data - filter out NaN and ensure valid numbers
  const chartData = (result.top_predictions || [])
    .filter(pred => {
      const prob = pred?.probability
      return prob !== null && prob !== undefined && !isNaN(prob) && isFinite(prob)
    })
    .map(pred => {
      const prob = Number(pred.probability) || 0
      return {
        name: String(pred.disease || 'Unknown').replace('_', ' '),
        probability: Math.max(0, Math.min(100, prob * 100)),
        color: prob >= 0.5 ? '#ef4444' : prob >= 0.3 ? '#f59e0b' : '#3b82f6'
      }
    })
    .filter(item => !isNaN(item.probability) && isFinite(item.probability))

  const allPredictionsData = Object.entries(result.predictions || {})
    .filter(([_, prob]) => {
      const p = Number(prob)
      return p !== null && p !== undefined && !isNaN(p) && isFinite(p)
    })
    .map(([disease, prob]) => {
      const p = Number(prob) || 0
      return {
        name: String(disease || 'Unknown').replace('_', ' '),
        probability: Math.max(0, Math.min(100, p * 100)),
        originalName: disease
      }
    })
    .filter(item => !isNaN(item.probability) && isFinite(item.probability))
    .sort((a, b) => {
      // Primary disease (Pneumonia) first
      const primaryDisease = result.primary_disease || result.top_disease || ''
      const secondaryDisease = result.secondary_disease || ''

      if (a.originalName === primaryDisease) return -1
      if (b.originalName === primaryDisease) return 1
      if (a.originalName === secondaryDisease) return -1
      if (b.originalName === secondaryDisease) return 1

      // Then sort by probability
      return b.probability - a.probability
    })

  return (
    <div className="max-w-7xl mx-auto px-4 py-12 space-y-8">
      {/* Diagnosis Cards - Primary & Secondary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Primary Disease */}
        <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-2xl shadow-2xl p-8 text-white">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-8 h-8" />
            <h2 className="text-xl font-bold">Primary Finding</h2>
          </div>
          <p className="text-4xl font-bold mb-2">
            {(result.primary_disease || result.top_disease || 'Unknown').replace('_', ' ')}
          </p>
          <div className="mt-4 bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 inline-block">
            <p className="text-sm font-semibold">DETECTED</p>
          </div>
        </div>

        {/* Secondary Disease */}
        {result.secondary_disease && (
          <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-2xl shadow-2xl p-8 text-white">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-8 h-8" />
              <h2 className="text-xl font-bold">Secondary Finding</h2>
            </div>
            <p className="text-4xl font-bold mb-2">
              {result.secondary_disease.replace('_', ' ')}
            </p>
            <div className="mt-4 bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 inline-block">
              <p className="text-sm font-semibold">DETECTED</p>
            </div>
          </div>
        )}
      </div>

      {/* All Diseases Status */}
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <CheckCircle className="w-6 h-6 text-primary-600" />
          <h3 className="text-2xl font-bold text-gray-800">Complete Analysis - All 14 Pathologies</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {allPredictionsData.map((pred, index) => {
            const isDetected = pred.probability > 50
            const isPrimary = pred.name === (result.primary_disease || result.top_disease || '').replace('_', ' ')
            const isSecondary = pred.name === (result.secondary_disease || '').replace('_', ' ')

            return (
              <div
                key={index}
                className={`border rounded-xl p-4 transition-all ${
                  isPrimary ? 'border-red-500 bg-red-50' :
                  isSecondary ? 'border-orange-500 bg-orange-50' :
                  isDetected ? 'border-yellow-500 bg-yellow-50' :
                  'border-gray-200 bg-white'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <span className="font-semibold text-gray-800">{pred.name}</span>
                    {isPrimary && <span className="ml-2 text-xs bg-red-500 text-white px-2 py-1 rounded">PRIMARY</span>}
                    {isSecondary && <span className="ml-2 text-xs bg-orange-500 text-white px-2 py-1 rounded">SECONDARY</span>}
                  </div>
                  <span className={`font-bold text-sm px-3 py-1 rounded ${
                    isDetected ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                  }`}>
                    {isDetected ? 'DETECTED' : 'Not Detected'}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Top Predictions Chart/Display - Hidden in mock mode */}
      {false && chartData.length > 0 && (
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-6 h-6 text-primary-600" />
            <h3 className="text-2xl font-bold text-gray-800">
              Top Prediction{chartData.length > 1 ? 's' : ''}
            </h3>
          </div>

          {chartData.length > 1 ? (
            <ResponsiveContainer width="100%" height={Math.max(100, chartData.length * 60)}>
              <BarChart data={chartData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  tickFormatter={(value) => `${value}%`}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={150}
                  style={{ fontSize: '14px' }}
                />
                <Tooltip
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: 'none',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Bar dataKey="probability" radius={[0, 8, 8, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="space-y-4">
              {chartData.map((pred, index) => (
                <div key={index} className="border border-gray-200 rounded-xl p-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xl font-bold text-gray-800">{pred.name}</span>
                    <span className="text-3xl font-bold" style={{ color: pred.color }}>
                      {pred.probability.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-4">
                    <div
                      className="h-4 rounded-full transition-all duration-500"
                      style={{
                        width: `${pred.probability}%`,
                        backgroundColor: pred.color
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

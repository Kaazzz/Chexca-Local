'use client'

import { AnalysisResult } from '../lib/api'
import { formatPercentage, getConfidenceColor, getConfidenceBgColor } from '../lib/utils'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { AlertTriangle, CheckCircle, TrendingUp, Download, FileText } from 'lucide-react'
import { generatePDF } from '../lib/pdfExport'
import { useState } from 'react'

interface ResultsPanelProps {
  result: AnalysisResult
}

export default function ResultsPanel({ result }: ResultsPanelProps) {
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false)

  // Get top 5 predictions from the model
  const topPredictions = result.top_predictions || []
  
  // Chart data for top predictions
  const chartData = topPredictions
    .map(pred => ({
      name: String(pred.disease || 'Unknown').replace('_', ' '),
      probability: Math.max(0, Math.min(100, pred.probability * 100)),
      color: pred.probability >= 0.5 ? '#ef4444' : pred.probability >= 0.3 ? '#f59e0b' : '#3b82f6'
    }))

  // All predictions sorted by probability
  const allPredictionsData = Object.entries(result.predictions || {})
    .map(([disease, prob]) => ({
      name: String(disease || 'Unknown').replace('_', ' '),
      probability: Math.max(0, Math.min(100, prob * 100)),
      originalName: disease,
      isDetected: prob > 0.5  // Threshold for detection
    }))
    .sort((a, b) => b.probability - a.probability)

  const topDisease = (result.top_disease || 'Unknown').replace('_', ' ')
  const topProbability = Number(result.top_disease_probability || 0) * 100

  const handleDownloadPDF = async () => {
    setIsGeneratingPDF(true)
    try {
      await generatePDF(result)
    } catch (error) {
      console.error('Error generating PDF:', error)
      alert('Failed to generate PDF. Please try again.')
    } finally {
      setIsGeneratingPDF(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-12 space-y-8" id="results-section">
      {/* Primary Finding Card */}
      <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-2xl shadow-2xl p-8 text-white">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle className="w-8 h-8" />
          <h2 className="text-xl font-bold">Primary Finding</h2>
        </div>
        <p className="text-4xl font-bold mb-2">{topDisease}</p>
        <p className="text-2xl font-semibold mb-4">Confidence: {topProbability.toFixed(1)}%</p>
        <div className="mt-4 bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 inline-block">
          <p className="text-sm font-semibold">
            {topProbability > 70 ? 'HIGH CONFIDENCE' : topProbability > 50 ? 'MODERATE CONFIDENCE' : 'LOW CONFIDENCE'}
          </p>
        </div>
      </div>

      {/* Top 5 Predictions Chart */}
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <TrendingUp className="w-6 h-6 text-primary-600" />
          <h3 className="text-2xl font-bold text-gray-800">Top 5 Predictions</h3>
        </div>

        {chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 60)}>
            <BarChart data={chartData} layout="vertical">
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
        )}
      </div>

      {/* All 14 Pathologies Status */}
      <div className="bg-white rounded-2xl shadow-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <CheckCircle className="w-6 h-6 text-primary-600" />
          <h3 className="text-2xl font-bold text-gray-800">Complete Analysis - All 14 Pathologies</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {allPredictionsData.map((pred, index) => {
            const isTop = index === 0

            return (
              <div
                key={index}
                className={`border rounded-xl p-4 transition-all ${
                  isTop ? 'border-red-500 bg-red-50' :
                  pred.isDetected ? 'border-orange-500 bg-orange-50' :
                  'border-gray-200 bg-white'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex-1">
                    <span className="font-semibold text-gray-800">{pred.name}</span>
                    {isTop && <span className="ml-2 text-xs bg-red-500 text-white px-2 py-1 rounded">PRIMARY</span>}
                  </div>
                  <span className={`font-bold text-lg ${
                    pred.isDetected ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {pred.probability.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      pred.isDetected ? 'bg-red-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${pred.probability}%` }}
                  ></div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Download PDF Button */}
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="text-center">
            <h3 className="text-xl font-bold text-gray-900 mb-2">Download Full Report</h3>
            <p className="text-gray-600">
              Get a comprehensive PDF summary of this analysis including all predictions, charts, and visual explanations.
            </p>
          </div>
          
          <button
            onClick={handleDownloadPDF}
            disabled={isGeneratingPDF}
            className={`
              flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-white
              transition-all duration-200 shadow-lg
              ${isGeneratingPDF 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 hover:shadow-xl hover:scale-105'
              }
            `}
          >
            {isGeneratingPDF ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Generating PDF...</span>
              </>
            ) : (
              <>
                <Download className="w-5 h-5" />
                <span>Download PDF Report</span>
                <FileText className="w-5 h-5" />
              </>
            )}
          </button>

          <p className="text-xs text-gray-500 text-center max-w-md">
            The report includes: Primary diagnosis, all disease probabilities, top predictions chart, 
            Grad-CAM heatmap visualization, and co-occurrence analysis.
          </p>
        </div>
      </div>
    </div>
  )
}

import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

interface AnalysisResult {
  predictions: Record<string, number>;
  top_predictions: Array<{ disease: string; probability: number }>;
  heatmap_overlay?: string;
  original_image?: string;
  top_disease: string;
  top_disease_probability: number;
  co_occurrence?: number[][];
  disease_classes: string[];
}

export async function generatePDF(result: AnalysisResult) {
  const pdf = new jsPDF('p', 'mm', 'a4');
  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 15;
  let yPosition = margin;

  // Helper function to add text
  const addText = (text: string, size: number = 10, isBold: boolean = false, extraSpacing: number = 0) => {
    pdf.setFontSize(size);
    pdf.setFont('helvetica', isBold ? 'bold' : 'normal');
    pdf.text(text, margin, yPosition);
    yPosition += size / 2 + 2 + extraSpacing;
  };

  // Helper function to check if we need a new page
  const checkNewPage = (heightNeeded: number = 20) => {
    if (yPosition + heightNeeded > pageHeight - margin) {
      pdf.addPage();
      yPosition = margin;
      return true;
    }
    return false;
  };

  // Title
  pdf.setFillColor(0, 147, 233);
  pdf.rect(0, 0, pageWidth, 30, 'F');
  pdf.setTextColor(255, 255, 255);
  pdf.setFontSize(20);
  pdf.setFont('helvetica', 'bold');
  pdf.text('CheXCA - Chest X-ray Analysis Report', pageWidth / 2, 20, { align: 'center' });
  
  yPosition = 40;
  pdf.setTextColor(0, 0, 0);

  // Date and Time
  const now = new Date();
  addText(`Generated: ${now.toLocaleString()}`, 9);
  yPosition += 8;

  // Primary Diagnosis Section
  addText('PRIMARY DIAGNOSIS', 14, true, 3);
  yPosition += 1;
  
  const topProbability = (result.top_disease_probability * 100).toFixed(1);
  addText(`${result.top_disease}`, 12, true);
  addText(`Confidence: ${topProbability}%`, 10);
  
  // Add confidence level indicator
  pdf.setFillColor(200, 200, 200);
  pdf.rect(margin, yPosition, pageWidth - 2 * margin, 8, 'F');
  
  const confidenceWidth = (pageWidth - 2 * margin) * result.top_disease_probability;
  let color = result.top_disease_probability > 0.7 ? [220, 38, 38] : 
              result.top_disease_probability > 0.5 ? [249, 115, 22] : 
              result.top_disease_probability > 0.3 ? [234, 179, 8] : [59, 130, 246];
  pdf.setFillColor(color[0], color[1], color[2]);
  pdf.rect(margin, yPosition, confidenceWidth, 8, 'F');
  yPosition += 18;

  checkNewPage();

  // Top 5 Predictions
  addText('TOP 5 PREDICTIONS', 14, true, 3);
  yPosition += 1;

  result.top_predictions.forEach((pred, index) => {
    checkNewPage(15);
    const probability = (pred.probability * 100).toFixed(1);
    addText(`${index + 1}. ${pred.disease}: ${probability}%`, 10);
  });
  yPosition += 10;

  checkNewPage(80);

  // All Disease Probabilities
  addText('ALL DISEASE PROBABILITIES', 14, true, 3);
  yPosition += 1;

  const sortedDiseases = Object.entries(result.predictions)
    .sort(([, a], [, b]) => b - a);

  sortedDiseases.forEach(([disease, probability]) => {
    checkNewPage(8);
    const prob = (probability * 100).toFixed(1);
    pdf.setFontSize(9);
    pdf.text(`${disease}:`, margin, yPosition);
    pdf.text(`${prob}%`, pageWidth - margin - 20, yPosition, { align: 'right' });
    yPosition += 5;
  });

  yPosition += 10;

  // Add Images Section
  if (result.heatmap_overlay || result.original_image) {
    checkNewPage(100);
    addText('VISUAL ANALYSIS', 14, true, 3);
    yPosition += 3;

    const imgWidth = (pageWidth - 3 * margin) / 2;
    const imgHeight = 70;

    try {
      if (result.original_image) {
        checkNewPage(imgHeight + 15);
        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Original X-ray', margin, yPosition);
        yPosition += 5;
        pdf.addImage(result.original_image, 'PNG', margin, yPosition, imgWidth, imgHeight);
      }

      if (result.heatmap_overlay) {
        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Grad-CAM Heatmap', result.original_image ? pageWidth / 2 + margin / 2 : margin, result.original_image ? yPosition - imgHeight - 5 : yPosition);
        if (!result.original_image) yPosition += 5;
        const xPos = result.original_image ? pageWidth / 2 + margin / 2 : margin;
        pdf.addImage(result.heatmap_overlay, 'PNG', xPos, yPosition, imgWidth, imgHeight);
      }

      yPosition += imgHeight + 10;
    } catch (error) {
      console.error('Error adding images to PDF:', error);
      addText('(Images could not be included)', 9);
    }
  }

  // Co-Occurrence Matrix Section
  if (result.co_occurrence && result.co_occurrence.length > 0 && result.disease_classes.length > 1) {
    checkNewPage(80);
    pdf.setFontSize(14);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(0, 0, 0);
    pdf.text('DISEASE CO-OCCURRENCE ANALYSIS', margin, yPosition);
    yPosition += 9;
    
    pdf.setFontSize(9);
    pdf.setFont('helvetica', 'normal');
    pdf.setTextColor(100, 100, 100);
    pdf.text('Correlation patterns between detected diseases', margin, yPosition);
    yPosition += 10;
    
    const matrix = result.co_occurrence;
    const classes = result.disease_classes;
    const cellSize = Math.min(12, (pageWidth - 2 * margin - 40) / classes.length);
    const matrixWidth = cellSize * classes.length;
    
    // Only show matrix if it fits reasonably
    if (classes.length <= 14) {
      // Color helper for matrix cells
      const getColorForValue = (value: number) => {
        const intensity = Math.min(value, 1);
        const r = Math.round(224 - (117 * intensity));
        const g = Math.round(212 - (179 * intensity));
        const b = Math.round(247 - (79 * intensity));
        return [r, g, b];
      };

      checkNewPage(matrixWidth + 50);
      
      // Draw matrix
      const startX = margin + 35;
      const startY = yPosition;
      
      // Draw cells
      for (let i = 0; i < classes.length; i++) {
        for (let j = 0; j < classes.length; j++) {
          const value = matrix[i]?.[j] ?? 0;
          const color = getColorForValue(value);
          
          pdf.setFillColor(color[0], color[1], color[2]);
          pdf.rect(startX + j * cellSize, startY + i * cellSize, cellSize, cellSize, 'F');
          pdf.setDrawColor(200, 200, 200);
          pdf.rect(startX + j * cellSize, startY + i * cellSize, cellSize, cellSize, 'S');
          
          // Add value text if cell is large enough
          if (cellSize >= 8) {
            pdf.setFontSize(6);
            pdf.setTextColor(60, 60, 60);
            const valueText = Math.round(value * 100).toString();
            pdf.text(valueText, startX + j * cellSize + cellSize / 2, startY + i * cellSize + cellSize / 2 + 1.5, { align: 'center' });
          }
        }
        
        // Row labels
        if (cellSize >= 6) {
          pdf.setFontSize(6);
          pdf.setTextColor(80, 80, 80);
          const label = classes[i].replace('_', ' ').substring(0, 12);
          pdf.text(label, startX - 2, startY + i * cellSize + cellSize / 2 + 1.5, { align: 'right' });
        }
      }
      
      yPosition += matrixWidth + 10;
      
      // Legend
      pdf.setFontSize(8);
      pdf.setTextColor(100, 100, 100);
      pdf.text('Values represent correlation strength (0-100%)', margin, yPosition);
      yPosition += 5;
      
      // Color scale
      const scaleWidth = 60;
      const scaleHeight = 5;
      for (let i = 0; i <= 10; i++) {
        const val = i / 10;
        const color = getColorForValue(val);
        pdf.setFillColor(color[0], color[1], color[2]);
        pdf.rect(margin + i * (scaleWidth / 10), yPosition, scaleWidth / 10, scaleHeight, 'F');
      }
      pdf.setDrawColor(150, 150, 150);
      pdf.rect(margin, yPosition, scaleWidth, scaleHeight, 'S');
      
      pdf.setFontSize(7);
      pdf.setTextColor(100, 100, 100);
      pdf.text('Low', margin, yPosition + scaleHeight + 3);
      pdf.text('High', margin + scaleWidth - 5, yPosition + scaleHeight + 3);
      
      yPosition += 15;
    } else {
      // Too many classes, just note it
      pdf.setFontSize(9);
      pdf.setTextColor(100, 100, 100);
      pdf.text(`Matrix available in web interface (${classes.length}×${classes.length})`, margin, yPosition);
      yPosition += 8;
    }
  }

  // Footer on last page
  pdf.setFontSize(8);
  pdf.setTextColor(128, 128, 128);
  pdf.text(
    'This report is for research and educational purposes only. Not for clinical diagnosis.',
    pageWidth / 2,
    pageHeight - 10,
    { align: 'center' }
  );

  // Generate filename with timestamp
  const filename = `CheXCA_Report_${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}.pdf`;
  
  // Save the PDF
  pdf.save(filename);
}

// Alternative: Capture the entire results section as an image
export async function generatePDFFromElement(elementId: string, filename: string = 'CheXCA_Report.pdf') {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error('Element not found:', elementId);
    return;
  }

  try {
    const canvas = await html2canvas(element, {
      scale: 2,
      useCORS: true,
      logging: false,
      backgroundColor: '#ffffff'
    });

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    
    const imgWidth = pageWidth - 20;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    
    let heightLeft = imgHeight;
    let position = 10;

    // Add first page
    pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
    heightLeft -= pageHeight;

    // Add additional pages if needed
    while (heightLeft > 0) {
      position = heightLeft - imgHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 10, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
    }

    const now = new Date();
    const timestamp = `${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}`;
    
    pdf.save(`${filename}_${timestamp}.pdf`);
  } catch (error) {
    console.error('Error generating PDF:', error);
    throw error;
  }
}

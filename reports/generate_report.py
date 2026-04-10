"""
PDF Report Generation Module for DeepFake Detection System
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime


class PDFReportGenerator:
    """Generate professional PDF reports for deepfake analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#16213e'),
            spaceBefore=12,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ContentText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))
    
    def generate_report(self, detection_result, user_info=None, heatmap_path=None, save_path='reports/analysis_report.pdf'):
        """
        Generate comprehensive PDF report
        
        Args:
            detection_result: Detection result dictionary
            user_info: User information (optional)
            heatmap_path: Path to heatmap image (optional)
            save_path: Path to save PDF report
            
        Returns:
            Path to generated report or None
        """
        try:
            # Ensure reports directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create document
            doc = SimpleDocTemplate(
                save_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Add content
            story.extend(self._create_title_section())
            story.append(Spacer(1, 0.3 * inch))
            
            if user_info:
                story.extend(self._create_user_section(user_info))
                story.append(Spacer(1, 0.2 * inch))
            
            story.extend(self._create_analysis_summary(detection_result))
            story.append(Spacer(1, 0.3 * inch))
            
            story.extend(self._create_detailed_results(detection_result))
            story.append(Spacer(1, 0.3 * inch))
            
            if heatmap_path and os.path.exists(heatmap_path):
                story.extend(self._create_heatmap_section(heatmap_path))
                story.append(Spacer(1, 0.3 * inch))
            
            story.extend(self._create_technical_details(detection_result))
            story.append(Spacer(1, 0.3 * inch))
            
            story.extend(self._create_disclaimer_section())
            
            # Build PDF
            doc.build(story)
            
            print(f"✓ Report generated: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"✗ Error generating report: {e}")
            return None
    
    def _create_title_section(self):
        """Create title section"""
        elements = []
        
        # Title
        title = Paragraph("DeepFake Detection Analysis Report", self.styles['CustomTitle'])
        elements.append(title)
        
        # Subtitle
        subtitle = Paragraph("Comprehensive Media Authentication Report", 
                           ParagraphStyle(
                               name='Subtitle',
                               parent=self.styles['Normal'],
                               fontSize=14,
                               textColor=colors.HexColor('#666666'),
                               alignment=TA_CENTER
                           ))
        elements.append(subtitle)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Date
        date_text = f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}"
        elements.append(Paragraph(date_text, self.styles['ContentText']))
        
        return elements
    
    def _create_user_section(self, user_info):
        """Create user information section"""
        elements = []
        
        elements.append(Paragraph("User Information", self.styles['SectionHeader']))
        
        user_data = [
            ['Username:', user_info.get('username', 'N/A')],
            ['Email:', user_info.get('email', 'N/A')],
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        user_table = Table(user_data, colWidths=[2*inch, 3*inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        elements.append(user_table)
        
        return elements
    
    def _create_analysis_summary(self, detection_result):
        """Create analysis summary section"""
        elements = []
        
        elements.append(Paragraph("Analysis Summary", self.styles['SectionHeader']))
        
        # Determine overall verdict
        prediction = detection_result.get('prediction', 'Unknown')
        confidence = detection_result.get('confidence', 0)
        
        if prediction == 'Fake':
            verdict = "CONTENT IDENTIFIED AS DEEPFAKE"
            verdict_color = colors.red
        else:
            verdict = "CONTENT VERIFIED AS AUTHENTIC"
            verdict_color = colors.green
        
        # Verdict box
        verdict_style = ParagraphStyle(
            name='Verdict',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=verdict_color,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        elements.append(Paragraph(verdict, verdict_style))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Key metrics
        confidence_pct = confidence * 100
        metrics_data = [
            ['Prediction', prediction],
            ['Confidence Score', f"{confidence_pct:.1f}%"],
            ['Real Probability', f"{detection_result.get('real_probability', 0) * 100:.1f}%"],
            ['Fake Probability', f"{detection_result.get('fake_probability', 0) * 100:.1f}%"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#fff4e6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc'))
        ]))
        
        elements.append(metrics_table)
        
        return elements
    
    def _create_detailed_results(self, detection_result):
        """Create detailed results section"""
        elements = []
        
        elements.append(Paragraph("Detailed Analysis Results", self.styles['SectionHeader']))
        
        # File information
        file_info = [
            ['File Name', detection_result.get('file_name', 'N/A')],
            ['File Type', detection_result.get('media_type', detection_result.get('file_type', 'N/A'))],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        if 'frames_analyzed' in detection_result:
            file_info.append(['Frames Analyzed', str(detection_result['frames_analyzed'])])
        
        if 'duration_analyzed' in detection_result:
            file_info.append(['Duration Analyzed', f"{detection_result['duration_analyzed']} seconds"])
        
        info_table = Table(file_info, colWidths=[2.5*inch, 2.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f9f9f9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Additional analysis details
        if 'anomaly_details' in detection_result:
            elements.append(Paragraph("Anomaly Detection", self.styles['SectionHeader']))
            anomaly_text = self._format_anomaly_details(detection_result['anomaly_details'])
            elements.append(Paragraph(anomaly_text, self.styles['ContentText']))
        
        return elements
    
    def _format_anomaly_details(self, anomaly_details):
        """Format anomaly details as HTML"""
        if isinstance(anomaly_details, dict):
            details_list = anomaly_details.get('details', [])
            if details_list:
                html = "<ul>"
                for detail in details_list:
                    html += f"<li>{detail.get('type', 'Unknown')}: {detail.get('severity', 'N/A')}</li>"
                html += "</ul>"
                return html
        return "No specific anomalies detected."
    
    def _create_heatmap_section(self, heatmap_path):
        """Create heatmap visualization section"""
        elements = []
        
        elements.append(Paragraph("Manipulation Heatmap Analysis", self.styles['SectionHeader']))
        
        description = Paragraph(
            "The heatmap below highlights regions with high probability of manipulation. "
            "Red areas indicate potential alterations, while green/blue areas suggest authentic content.",
            self.styles['ContentText']
        )
        elements.append(description)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Add heatmap image
        try:
            img = RLImage(heatmap_path, width=6*inch, height=4*inch)
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"Heatmap image not available: {e}", self.styles['ContentText']))
        
        return elements
    
    def _create_technical_details(self, detection_result):
        """Create technical details section"""
        elements = []
        
        elements.append(Paragraph("Technical Methodology", self.styles['SectionHeader']))
        
        tech_text = """
        This analysis employs state-of-the-art deep learning techniques for deepfake detection:
        <br/><br/>
        <b>Image Analysis:</b> Convolutional Neural Networks (CNN) with ResNet50 architecture trained on FaceForensics++ and CelebDF datasets.
        <br/><br/>
        <b>Video Analysis:</b> Combined CNN and Long Short-Term Memory (LSTM) networks for spatiotemporal feature extraction.
        <br/><br/>
        <b>Audio Analysis:</b> Mel spectrogram-based CNN analysis for detecting synthetic voice patterns.
        <br/><br/>
        <b>Facial Analysis:</b> Multi-point facial landmark detection, eye blink pattern analysis, and lip-sync verification.
        <br/><br/>
        <b>Accuracy:</b> Our models achieve 90%+ accuracy on benchmark datasets when properly trained.
        """
        
        elements.append(Paragraph(tech_text, self.styles['ContentText']))
        
        return elements
    
    def _create_disclaimer_section(self):
        """Create disclaimer section"""
        elements = []
        
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("Disclaimer", self.styles['SectionHeader']))
        
        disclaimer_text = """
        This report is generated by an automated AI system for informational purposes only. 
        While our detection systems employ advanced machine learning techniques, no system is infallible. 
        The results should be considered as one piece of evidence in a broader verification process.
        <br/><br/>
        This report does not constitute definitive proof of authenticity or manipulation. 
        For critical applications, please consult with digital forensics experts and use multiple verification methods.
        <br/><br/>
        <b>Confidence Level:</b> Always consider the confidence score when interpreting results. 
        Higher confidence (>80%) indicates more reliable predictions.
        """
        
        disclaimer_style = ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Footer
        footer_text = f"Report generated by DeepFake Detection System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        footer_style = ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#999999'),
            alignment=TA_CENTER
        )
        
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(footer_text, footer_style))
        
        return elements
    
    def generate_simple_report(self, detection_result, save_path='reports/simple_report.pdf'):
        """
        Generate a simplified one-page report
        
        Args:
            detection_result: Detection result dictionary
            save_path: Path to save PDF
            
        Returns:
            Path to generated report
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            doc = SimpleDocTemplate(save_path, pagesize=letter)
            story = []
            
            # Quick summary
            story.extend(self._create_title_section())
            story.append(Spacer(1, 0.2 * inch))
            story.extend(self._create_analysis_summary(detection_result))
            
            doc.build(story)
            
            print(f"✓ Simple report generated: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"✗ Error generating simple report: {e}")
            return None


def test_report_generation():
    """Test PDF report generation"""
    generator = PDFReportGenerator()
    
    # Sample detection result
    test_result = {
        'prediction': 'Fake',
        'confidence': 0.92,
        'is_fake': True,
        'file_name': 'test_video.mp4',
        'file_type': 'video',
        'frames_analyzed': 150,
        'real_probability': 0.08,
        'fake_probability': 0.92,
        'anomaly_details': {
            'has_anomalies': True,
            'details': [
                {'type': 'abnormal_blink_rate', 'severity': 'medium'},
                {'type': 'lip_sync_mismatch', 'severity': 'high'}
            ]
        }
    }
    
    user_info = {
        'username': 'testuser',
        'email': 'test@example.com'
    }
    
    # Generate report
    report_path = generator.generate_report(test_result, user_info=user_info)
    
    print(f"\nTest report generated at: {report_path}")
    
    return generator


if __name__ == "__main__":
    generator = test_report_generation()

from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import os
from datetime import datetime
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Import our AI modules
try:
    from disease_model import detector
    print("✅ Disease detection module loaded successfully")
except Exception as e:
    print(f"❌ Error loading disease_model: {e}")
    detector = None

try:
    from monitor import monitor
    print("✅ Monitor module loaded successfully")
except Exception as e:
    print(f"❌ Error loading monitor: {e}")
    monitor = None

# Create Flask app FIRST - This is CRITICAL
app = Flask(__name__)

# Database initialization
def init_db():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        # Create prescriptions table
        c.execute('''CREATE TABLE IF NOT EXISTS prescriptions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      patient_name TEXT,
                      age INTEGER,
                      symptoms TEXT,
                      medicines TEXT,
                      doctor_notes TEXT,
                      vitals TEXT,
                      created_at TIMESTAMP)''')
        
        # Create monitoring logs table
        c.execute('''CREATE TABLE IF NOT EXISTS monitoring_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      prescription_id INTEGER,
                      risk_score REAL,
                      risk_level TEXT,
                      alerts TEXT,
                      created_at TIMESTAMP)''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# AI Engine - Rule-based medicine suggestions
def get_ai_suggestions(symptoms_list):
    """AI Engine: Suggests medicines based on symptoms"""
    suggestions = []
    
    medicine_rules = {
        "fever": [
            {"medicine": "Paracetamol", "dosage": "500mg", "timing": "Morning/Night", "food": "After food", "category": "Analgesic"},
            {"medicine": "Ibuprofen", "dosage": "400mg", "timing": "After meals", "food": "After food", "category": "NSAID"}
        ],
        "cough": [
            {"medicine": "Dextromethorphan", "dosage": "15mg", "timing": "3 times daily", "food": "After food", "category": "Antitussive"},
            {"medicine": "Guaifenesin", "dosage": "200mg", "timing": "Every 4 hours", "food": "With water", "category": "Expectorant"}
        ],
        "cold": [
            {"medicine": "Cetirizine", "dosage": "10mg", "timing": "Night", "food": "Before sleep", "category": "Antihistamine"},
            {"medicine": "Pseudoephedrine", "dosage": "60mg", "timing": "Every 6 hours", "food": "With food", "category": "Decongestant"}
        ],
        "headache": [
            {"medicine": "Aspirin", "dosage": "300mg", "timing": "As needed", "food": "After food", "category": "Analgesic"},
            {"medicine": "Naproxen", "dosage": "250mg", "timing": "Twice daily", "food": "With food", "category": "NSAID"}
        ],
        "diabetes": [
            {"medicine": "Metformin", "dosage": "500mg", "timing": "Twice daily", "food": "With meals", "category": "Antidiabetic"},
            {"medicine": "Glimepiride", "dosage": "2mg", "timing": "Morning", "food": "Before breakfast", "category": "Sulfonylurea"}
        ],
        "hypertension": [
            {"medicine": "Amlodipine", "dosage": "5mg", "timing": "Morning", "food": "With or without food", "category": "CCB"},
            {"medicine": "Lisinopril", "dosage": "10mg", "timing": "Morning", "food": "Before food", "category": "ACE Inhibitor"}
        ],
        "infection": [
            {"medicine": "Amoxicillin", "dosage": "500mg", "timing": "3 times daily", "food": "Before food", "category": "Antibiotic"},
            {"medicine": "Ciprofloxacin", "dosage": "500mg", "timing": "Twice daily", "food": "Empty stomach", "category": "Antibiotic"}
        ],
        "skin_infection": [
            {"medicine": "Clotrimazole Cream", "dosage": "Apply thin layer", "timing": "Twice daily", "food": "External use", "category": "Antifungal"},
            {"medicine": "Mupirocin Ointment", "dosage": "Apply 3 times", "timing": "Three times daily", "food": "External use", "category": "Antibiotic"}
        ]
    }
    
    for symptom in symptoms_list:
        symptom_lower = symptom.lower().strip()
        if symptom_lower in medicine_rules:
            for med in medicine_rules[symptom_lower]:
                if med not in suggestions:
                    suggestions.append(med)
    
    return suggestions

# Safety check for missing medications
def check_missing_medicines(symptoms_list, current_medicines):
    """Check for potentially missing essential medications"""
    alerts = []
    current_med_names = [m['medicine'].lower() for m in current_medicines]
    
    has_nsaid = any('ibuprofen' in name or 'aspirin' in name or 'naproxen' in name 
                    for name in current_med_names)
    
    if has_nsaid and 'gastric' not in str(current_med_names).lower():
        alerts.append({
            "type": "warning",
            "message": " NSAID detected. Consider adding gastric protection (e.g., Omeprazole 20mg)"
        })
    
    has_antibiotic = any('amoxicillin' in name or 'ciprofloxacin' in name 
                         for name in current_med_names)
    
    if has_antibiotic:
        alerts.append({
            "type": "info",
            "message": "💡 Antibiotic prescribed. Consider adding probiotics"
        })
    
    return alerts

# ROUTES - All routes come AFTER app is created
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/suggest', methods=['POST'])
def suggest_medicines():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        
        suggestions = get_ai_suggestions(symptoms)
        alerts = check_missing_medicines(symptoms, suggestions)
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'alerts': alerts
        })
    except Exception as e:
        print(f"Error in suggest_medicines: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect_disease', methods=['POST'])
def detect_disease():
    """AI Skin Disease Detection"""
    try:
        if detector is None:
            return jsonify({
                'success': False, 
                'error': 'Disease detection module not available. Please check installation.'
            }), 500
        
        data = request.json
        image_base64 = data.get('image', '')
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Validate image size (max 10MB)
        if len(image_base64) > 10 * 1024 * 1024:
            return jsonify({
                'success': False, 
                'error': 'Image too large. Please upload image less than 10MB'
            }), 400
        
        print("🔍 Analyzing skin image...")
        result = detector.analyze_skin_image(image_base64)
        
        # Add fallback for very low confidence
        if result.get('success', False) and result.get('confidence', 0) < 30:
            result['warning'] = "⚠️ Low confidence detection. Please consult a dermatologist for confirmation."
        
        print(f"✅ Disease detection complete: {result.get('disease', 'Unknown')}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Disease detection error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Analysis failed: {str(e)}',
            'suggestion': 'Please try with a clearer image or consult a doctor directly.'
        }), 500

@app.route('/api/monitor_prescription', methods=['POST'])
def monitor_prescription():
    """AI Monitor System"""
    try:
        if monitor is None:
            return jsonify({
                'success': False,
                'error': 'Monitor module not available'
            }), 500
            
        data = request.json
        monitoring_result = monitor.monitor_prescription(data)
        
        # Save to database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('''INSERT INTO monitoring_logs 
                     (prescription_id, risk_score, risk_level, alerts, created_at)
                     VALUES (?, ?, ?, ?, ?)''',
                  (data.get('prescription_id', 0),
                   monitoring_result['risk_score'],
                   monitoring_result['risk_level'],
                   json.dumps(monitoring_result['alerts']),
                   datetime.now()))
        conn.commit()
        conn.close()
        
        return jsonify(monitoring_result)
    except Exception as e:
        print(f"Error in monitor_prescription: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save_prescription', methods=['POST'])
def save_prescription():
    try:
        data = request.json
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO prescriptions 
                     (patient_name, age, symptoms, medicines, doctor_notes, vitals, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (data['patient_name'], data['age'], 
                   json.dumps(data['symptoms']), 
                   json.dumps(data['medicines']),
                   data.get('doctor_notes', ''),
                   json.dumps(data.get('vitals', {})),
                   datetime.now()))
        
        conn.commit()
        prescription_id = c.lastrowid
        conn.close()
        
        return jsonify({
            'success': True,
            'prescription_id': prescription_id
        })
    except Exception as e:
        print(f"Error in save_prescription: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3c72'),
            alignment=1,
            spaceAfter=30
        )
        story.append(Paragraph("Medical Prescription", title_style))
        
        hospital_style = ParagraphStyle(
            'HospitalStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=1
        )
        story.append(Paragraph("AI-Powered Prescription Assistant", hospital_style))
        story.append(Spacer(1, 20))
        
        patient_info = [
            ["Patient Name:", data['patient_name']],
            ["Age:", str(data['age'])],
            ["Date:", datetime.now().strftime("%d/%m/%Y")]
        ]
        
        patient_table = Table(patient_info, colWidths=[100, 300])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3c72')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        med_data = [["Medicine", "Dosage", "Timing", "Instructions"]]
        for med in data['medicines']:
            med_data.append([
                med['medicine'],
                med['dosage'],
                med['timing'],
                med['food']
            ])
        
        med_table = Table(med_data, colWidths=[120, 80, 100, 150])
        med_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3c72')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(med_table)
        story.append(Spacer(1, 20))
        
        if data.get('doctor_notes'):
            story.append(Paragraph("Doctor's Notes:", styles['Heading4']))
            story.append(Paragraph(data['doctor_notes'], styles['Normal']))
            story.append(Spacer(1, 20))
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1
        )
        story.append(Paragraph("This is an AI-assisted prescription. Doctor has reviewed and approved.", footer_style))
        story.append(Paragraph("For any questions, please consult your doctor.", footer_style))
        
        doc.build(story)
        buffer.seek(0)
        
        pdf_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'pdf': pdf_base64
        })
    except Exception as e:
        print(f"Error in generate_pdf: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT * FROM prescriptions ORDER BY created_at DESC LIMIT 50')
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'patient_name': row[1],
                'age': row[2],
                'symptoms': json.loads(row[3]),
                'medicines': json.loads(row[4]),
                'doctor_notes': row[5],
                'created_at': row[6]
            })
        
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error in get_history: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Run the app - This should be at the end
if __name__ == '__main__':
    init_db()
    print("🚀 Starting AI Prescription Assistant...")
    print("📍 Server will run at: http://111.0.0.1:50")
    app.run(debug=True, port=5000)
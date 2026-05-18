import sqlite3
import json
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

class AIMonitor:
    """AI System to monitor prescriptions and patient health"""
    
    def __init__(self):
        self.alert_thresholds = {
            'drug_interaction_severity': 0.7,
            'medication_completeness': 0.8,
            'adherence_risk': 0.6
        }
    
    def monitor_prescription(self, prescription_data):
        """Monitor prescription for potential issues"""
        alerts = []
        recommendations = []
        risk_score = 0
        
        # Check for drug interactions
        interactions = self.check_drug_interactions(prescription_data['medicines'])
        if interactions:
            risk_score += 0.3
            alerts.extend(interactions)
        
        # Check completeness
        completeness = self.check_prescription_completeness(prescription_data)
        if completeness['score'] < self.alert_thresholds['medication_completeness']:
            alerts.append({
                'type': 'warning',
                'message': f"Incomplete prescription: {completeness['missing_items']}"
            })
            risk_score += 0.2
        
        # Check dosage appropriateness
        dosage_issues = self.check_dosage_appropriateness(prescription_data['medicines'])
        if dosage_issues:
            alerts.extend(dosage_issues)
            risk_score += 0.2
        
        # Generate recommendations
        recommendations = self.generate_recommendations(prescription_data)
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_level': self.get_risk_level(risk_score),
            'alerts': alerts,
            'recommendations': recommendations,
            'quality_score': completeness['score']
        }
    
    def check_drug_interactions(self, medicines):
        """Check for potential drug interactions"""
        interactions = []
        medicine_names = [m['medicine'].lower() for m in medicines]
        
        # Known interaction pairs
        interaction_pairs = [
            (['ibuprofen', 'aspirin'], 'Increased risk of stomach bleeding'),
            (['amoxicillin', 'ciprofloxacin'], 'Reduced antibiotic effectiveness'),
            (['metformin', 'glimepiride'], 'Increased risk of hypoglycemia'),
            (['amlodipine', 'lisinopril'], 'Enhanced blood pressure lowering effect'),
            (['paracetamol', 'ibuprofen'], 'Increased risk of liver/kidney damage'),
            (['warfarin', 'aspirin'], 'Increased bleeding risk'),
            (['lisinopril', 'spironolactone'], 'Increased potassium levels')
        ]
        
        for pair, message in interaction_pairs:
            if all(drug in medicine_names for drug in pair):
                interactions.append({
                    'type': 'interaction',
                    'severity': 'moderate',
                    'message': f"⚠️ Drug interaction detected: {message}"
                })
        
        return interactions
    
    def check_prescription_completeness(self, prescription):
        """Check if prescription is complete"""
        missing = []
        score = 1.0
        
        medicines = prescription['medicines']
        
        # Check if dosage specified
        for med in medicines:
            if not med.get('dosage') or med['dosage'] == '':
                missing.append(f"Dosage missing for {med['medicine']}")
                score -= 0.1
        
        # Check if timing specified
        for med in medicines:
            if not med.get('timing') or med['timing'] == '':
                missing.append(f"Timing missing for {med['medicine']}")
                score -= 0.1
        
        # Check for gastric protection with NSAIDs
        has_nsaid = any('ibuprofen' in m['medicine'].lower() or 'aspirin' in m['medicine'].lower() 
                       or 'naproxen' in m['medicine'].lower()
                       for m in medicines)
        has_gastric = any('gastric' in m['medicine'].lower() or 'omeprazole' in m['medicine'].lower()
                         or 'pantoprazole' in m['medicine'].lower()
                         for m in medicines)
        
        if has_nsaid and not has_gastric:
            missing.append("Gastric protection recommended with NSAIDs")
            score -= 0.1
        
        # Check for number of medicines
        if len(medicines) > 5:
            missing.append("Multiple medicines prescribed - check for interactions")
            score -= 0.05
        
        return {
            'score': max(score, 0),
            'missing_items': missing
        }
    
    def check_dosage_appropriateness(self, medicines):
        """Check if dosages are appropriate"""
        issues = []
        
        dosage_ranges = {
            'paracetamol': (500, 1000),
            'ibuprofen': (200, 600),
            'amoxicillin': (250, 500),
            'cetirizine': (5, 10),
            'metformin': (500, 2000),
            'aspirin': (300, 600),
            'amlodipine': (5, 10),
            'lisinopril': (10, 40)
        }
        
        for med in medicines:
            med_name = med['medicine'].lower()
            if med_name in dosage_ranges and med.get('dosage'):
                try:
                    # Extract numeric value from dosage string
                    import re
                    numbers = re.findall(r'\d+', med['dosage'])
                    if numbers:
                        dosage = int(numbers[0])
                        min_dose, max_dose = dosage_ranges[med_name]
                        if dosage < min_dose:
                            issues.append({
                                'type': 'warning',
                                'message': f"{med['medicine']} dosage ({dosage}mg) below recommended minimum ({min_dose}mg)"
                            })
                        elif dosage > max_dose:
                            issues.append({
                                'type': 'critical',
                                'message': f"{med['medicine']} dosage ({dosage}mg) exceeds maximum ({max_dose}mg)"
                            })
                except:
                    pass
        
        return issues
    
    def generate_recommendations(self, prescription):
        """Generate AI recommendations"""
        recommendations = []
        
        medicines = prescription['medicines']
        patient_age = prescription.get('age', 0)
        
        # Elderly patient considerations
        if patient_age > 65:
            recommendations.append("👴 Elderly patient detected - Consider lower starting doses and monitor for side effects")
        
        if patient_age < 12:
            recommendations.append("👶 Pediatric patient - Verify all doses are weight-appropriate")
        
        # Check for antibiotics
        has_antibiotic = any('cillin' in m['medicine'].lower() or 'floxacin' in m['medicine'].lower()
                            or 'mycin' in m['medicine'].lower()
                            for m in medicines)
        if has_antibiotic:
            recommendations.append("💊 Complete full course of antibiotics even if symptoms improve")
            recommendations.append("🦠 Consider probiotics to maintain gut health")
        
        # Check for painkillers
        has_painkiller = any('paracetamol' in m['medicine'].lower() or 'ibuprofen' in m['medicine'].lower()
                            or 'aspirin' in m['medicine'].lower()
                            for m in medicines)
        if has_painkiller:
            recommendations.append("💡 Maximum paracetamol: 4g/day. Avoid exceeding recommended doses")
        
        # Check for multiple medications
        if len(medicines) > 3:
            recommendations.append("📋 Multiple medications prescribed - Review for potential interactions")
        
        # General recommendations
        recommendations.append("📱 Set medication reminders for better adherence")
        recommendations.append("💧 Stay hydrated and maintain proper nutrition")
        recommendations.append("🩺 Schedule follow-up visit to monitor treatment effectiveness")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_risk_level(self, score):
        """Determine risk level based on score"""
        if score < 0.3:
            return "Low Risk ✅"
        elif score < 0.6:
            return "Medium Risk ⚠️"
        else:
            return "High Risk ❌ - Immediate Review Needed"
    
    def monitor_vitals(self, vitals_data):
        """Monitor patient vitals"""
        alerts = []
        
        # Check blood pressure
        if 'bp_systolic' in vitals_data:
            if vitals_data['bp_systolic'] > 140:
                alerts.append("⚠️ High blood pressure detected - Monitor and consider medication adjustment")
            elif vitals_data['bp_systolic'] < 90:
                alerts.append("⚠️ Low blood pressure detected - Risk of dizziness")
        
        # Check temperature
        if 'temperature' in vitals_data:
            if vitals_data['temperature'] > 38:
                alerts.append("⚠️ Fever detected - Monitor for infection")
            elif vitals_data['temperature'] < 36:
                alerts.append("⚠️ Low body temperature - Keep patient warm")
        
        # Check blood sugar
        if 'blood_sugar' in vitals_data:
            if vitals_data['blood_sugar'] > 180:
                alerts.append("⚠️ High blood sugar detected - Consider adjusting diabetes medication")
            elif vitals_data['blood_sugar'] < 70:
                alerts.append("⚠️ Low blood sugar detected - Immediate attention needed")
        
        # Check heart rate
        if 'heart_rate' in vitals_data:
            if vitals_data['heart_rate'] > 100:
                alerts.append("⚠️ Tachycardia detected - Monitor cardiac status")
            elif vitals_data['heart_rate'] < 60:
                alerts.append("⚠️ Bradycardia detected - Review medications")
        
        return alerts

# Global monitor instance
monitor = AIMonitor()
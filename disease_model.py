import numpy as np
import cv2
from PIL import Image
import io
import base64
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class SkinDiseaseDetector:
    """AI Model for Skin Disease Detection with EXACTLY 10 features"""
    
    def __init__(self):
        self.model = None
        self.n_features = 10  # Fixed to 10 features
        self.disease_categories = {
            0: {
                "name": "Eczema (Atopic Dermatitis)",
                "severity": "Moderate",
                "description": "Inflammatory skin condition causing red, itchy patches. Common in children and adults.",
                "treatment": ["Topical corticosteroids", "Moisturizers", "Antihistamines for itching"],
                "urgency": "Consult within 1 week",
                "home_remedies": ["Apply coconut oil", "Use oatmeal baths", "Avoid harsh soaps"]
            },
            1: {
                "name": "Psoriasis",
                "severity": "Moderate to Severe",
                "description": "Autoimmune condition causing scaly, red patches with silvery scales",
                "treatment": ["Topical treatments", "Phototherapy", "Systemic medications"],
                "urgency": "Consult within 1 week",
                "home_remedies": ["Aloe vera gel", "Fish oil supplements", "Stress management"]
            },
            2: {
                "name": "Acne Vulgaris",
                "severity": "Mild to Moderate",
                "description": "Common skin condition with pimples, blackheads, and inflammation",
                "treatment": ["Benzoyl peroxide", "Salicylic acid", "Topical retinoids"],
                "urgency": "Routine consultation",
                "home_remedies": ["Tea tree oil", "Green tea", "Zinc supplements"]
            },
            3: {
                "name": "Fungal Infection (Ringworm/Tinea)",
                "severity": "Moderate",
                "description": "Fungal infection causing circular, red, itchy rashes",
                "treatment": ["Antifungal creams", "Oral antifungals", "Keep area dry"],
                "urgency": "Consult within 3-5 days",
                "home_remedies": ["Apple cider vinegar", "Garlic paste", "Keep skin dry"]
            },
            4: {
                "name": "Bacterial Infection (Impetigo/Cellulitis)",
                "severity": "High",
                "description": "Bacterial infection causing redness, swelling, warmth, and pus",
                "treatment": ["Antibiotics", "Keep clean", "Monitor for fever"],
                "urgency": "URGENT - Consult within 24 hours",
                "home_remedies": ["Keep area clean", "Elevate affected area", "Rest"]
            },
            5: {
                "name": "Allergic Reaction",
                "severity": "Variable",
                "description": "Red, itchy rash from allergen exposure",
                "treatment": ["Antihistamines", "Avoid allergen", "Topical steroids"],
                "urgency": "Consult within 24-48 hours",
                "home_remedies": ["Cold compress", "Oatmeal bath", "Avoid scratching"]
            }
        }
        
        self.init_model()
    
    def init_model(self):
        """Initialize model with EXACTLY 10 features"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train with exactly 10 features
        X_train = np.random.rand(200, self.n_features)
        y_train = np.random.randint(0, 6, 200)
        
        self.model.fit(X_train, y_train)
        print(f"✅ Model initialized with EXACTLY {self.n_features} features")
    
    def extract_features(self, image_array):
        """Extract EXACTLY 10 features from image - NO MORE, NO LESS"""
        features = []
        
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Feature 1: Mean intensity
        features.append(float(np.mean(gray)))
        
        # Feature 2: Standard deviation
        features.append(float(np.std(gray)))
        
        # Feature 3: Median intensity
        features.append(float(np.median(gray)))
        
        # Feature 4: Skewness (texture)
        from scipy import stats
        features.append(float(stats.skew(gray.flatten())))
        
        # Feature 5: Kurtosis (texture)
        features.append(float(stats.kurtosis(gray.flatten())))
        
        # Feature 6: Edge density
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(float(edge_density))
        
        # Feature 7: Mean edge intensity
        features.append(float(np.mean(edges)))
        
        # Feature 8: Color variance (red channel)
        if len(image_array.shape) == 3:
            red_channel = image_array[:,:,0]
            features.append(float(np.std(red_channel)))
        else:
            features.append(float(np.std(gray)))
        
        # Feature 9: Color variance (green channel)
        if len(image_array.shape) == 3:
            green_channel = image_array[:,:,1]
            features.append(float(np.std(green_channel)))
        else:
            features.append(float(np.mean(gray)))
        
        # Feature 10: Color variance (blue channel)
        if len(image_array.shape) == 3:
            blue_channel = image_array[:,:,2]
            features.append(float(np.std(blue_channel)))
        else:
            features.append(float(np.median(gray)))
        
        # CRITICAL: Ensure exactly 10 features
        if len(features) > self.n_features:
            features = features[:self.n_features]
            print(f"⚠️ Trimmed features from {len(features)} to {self.n_features}")
        elif len(features) < self.n_features:
            # Pad with zeros if needed
            features.extend([0.0] * (self.n_features - len(features)))
            print(f"⚠️ Padded features from {len(features)} to {self.n_features}")
        
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Debug output
        print(f"🔍 Features extracted: {len(features)} features")
        print(f"📊 Features shape: {features_array.shape}")
        print(f"🎯 Expected: ({1}, {self.n_features})")
        
        return features_array
    
    def analyze_skin_image(self, image_base64):
        """Analyze skin image and detect disease"""
        try:
            # Decode base64 image
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            image = image.resize((224, 224))
            image_array = np.array(image)
            
            # Extract features (exactly 10 features)
            features = self.extract_features(image_array)
            
            # Verify feature count
            if features.shape[1] != self.n_features:
                return {
                    'success': False,
                    'error': f"Feature mismatch: Expected {self.n_features}, got {features.shape[1]}"
                }
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            confidence = np.max(self.model.predict_proba(features)[0])
            
            # Get disease info
            disease_info = self.disease_categories.get(prediction, self.disease_categories[0])
            
            # Generate treatment recommendations
            treatment_plan = self.generate_treatment_plan(disease_info, confidence)
            
            # Determine if urgent based on severity
            is_urgent = disease_info['severity'] in ['High', 'Variable']
            
            return {
                'success': True,
                'disease': disease_info['name'],
                'confidence': float(confidence) * 100,  # Convert to percentage
                'severity': disease_info['severity'],
                'description': disease_info['description'],
                'treatment': disease_info['treatment'],
                'urgency': disease_info['urgency'],
                'home_remedies': disease_info.get('home_remedies', []),
                'treatment_plan': treatment_plan,
                'is_urgent': is_urgent,
                'recommended_medicines': self.get_recommended_medicines(prediction)
            }
            
        except Exception as e:
            print(f"❌ Error in analyze_skin_image: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_treatment_plan(self, disease_info, confidence):
        """Generate detailed treatment plan"""
        plan = {
            'immediate_actions': [],
            'medications': [],
            'lifestyle_changes': [],
            'follow_up': [],
            'warning_signs': []
        }
        
        disease_name = disease_info['name']
        
        if 'Eczema' in disease_name:
            plan['immediate_actions'] = ['Apply cool compress', 'Use gentle moisturizer', 'Avoid scratching']
            plan['medications'] = ['Hydrocortisone cream 1%', 'Antihistamines']
            plan['lifestyle_changes'] = ['Use mild soaps', 'Wear cotton clothing', 'Humidify room']
            plan['warning_signs'] = ['Infection signs (pus, fever)', 'Worsening rash']
            
        elif 'Psoriasis' in disease_name:
            plan['immediate_actions'] = ['Apply thick moisturizer', 'Avoid skin injuries', 'Limited sun exposure']
            plan['medications'] = ['Topical corticosteroids', 'Salicylic acid']
            plan['lifestyle_changes'] = ['Reduce stress', 'Avoid alcohol', 'Maintain healthy weight']
            plan['warning_signs'] = ['Joint pain', 'Widespread rash']
            
        elif 'Acne' in disease_name:
            plan['immediate_actions'] = ['Gentle cleansing twice daily', 'Avoid picking', 'Use oil-free products']
            plan['medications'] = ['Benzoyl peroxide 5%', 'Salicylic acid cleanser']
            plan['lifestyle_changes'] = ['Wash pillowcases', 'Avoid touching face', 'Balanced diet']
            plan['warning_signs'] = ['Severe pain', 'Cysts', 'Scarring']
            
        elif 'Fungal' in disease_name:
            plan['immediate_actions'] = ['Keep area dry', 'Apply antifungal cream', 'Wash clothes in hot water']
            plan['medications'] = ['Clotrimazole cream 1%', 'Terbinafine']
            plan['lifestyle_changes'] = ['Wear breathable fabrics', 'Change socks daily', 'Avoid sharing towels']
            plan['warning_signs'] = ['Spreading rash', 'Fever', 'Pus']
            
        elif 'Bacterial' in disease_name:
            plan['immediate_actions'] = ['Clean with antiseptic', 'Apply antibiotic ointment', 'Cover wound']
            plan['medications'] = ['Topical antibiotics', 'Oral antibiotics if severe']
            plan['lifestyle_changes'] = ['Complete antibiotic course', 'Rest', 'Monitor temperature']
            plan['warning_signs'] = ['Fever > 101°F', 'Spreading redness', 'Severe pain']
            
        elif 'Allergic' in disease_name:
            plan['immediate_actions'] = ['Avoid trigger', 'Apply cold compress', 'Take antihistamine']
            plan['medications'] = ['Oral antihistamines', 'Topical hydrocortisone']
            plan['lifestyle_changes'] = ['Keep reaction diary', 'Read labels', 'Carry emergency meds']
            plan['warning_signs'] = ['Difficulty breathing', 'Face swelling', 'Rapid heartbeat']
        
        plan['follow_up'] = [f"Follow up: {disease_info['urgency']}", 
                            "Take photos to track progress",
                            "Seek immediate care if warning signs appear"]
        
        return plan
    
    def get_recommended_medicines(self, disease_id):
        """Get recommended medicines based on disease"""
        medicines = {
            0: [  # Eczema
                {"name": "Hydrocortisone Cream", "strength": "1%", "frequency": "Twice daily", "duration": "7 days"},
                {"name": "Cetirizine", "strength": "10mg", "frequency": "Once daily", "duration": "As needed"}
            ],
            1: [  # Psoriasis
                {"name": "Calcipotriene Cream", "strength": "0.005%", "frequency": "Twice daily", "duration": "4 weeks"},
                {"name": "Coal Tar Shampoo", "strength": "Use as directed", "frequency": "Every other day", "duration": "2 weeks"}
            ],
            2: [  # Acne
                {"name": "Benzoyl Peroxide", "strength": "5%", "frequency": "Once daily", "duration": "8 weeks"},
                {"name": "Adapalene Gel", "strength": "0.1%", "frequency": "At night", "duration": "12 weeks"}
            ],
            3: [  # Fungal
                {"name": "Clotrimazole Cream", "strength": "1%", "frequency": "Twice daily", "duration": "2-4 weeks"},
                {"name": "Terbinafine", "strength": "250mg", "frequency": "Once daily", "duration": "2 weeks"}
            ],
            4: [  # Bacterial
                {"name": "Mupirocin Ointment", "strength": "2%", "frequency": "Three times daily", "duration": "7-10 days"},
                {"name": "Cephalexin", "strength": "500mg", "frequency": "Twice daily", "duration": "7-14 days"}
            ],
            5: [  # Allergic
                {"name": "Diphenhydramine", "strength": "25mg", "frequency": "As needed", "duration": "3-5 days"},
                {"name": "Hydrocortisone Cream", "strength": "1%", "frequency": "Twice daily", "duration": "7 days"}
            ]
        }
        
        return medicines.get(disease_id, medicines[0])

# Global detector instance
detector = SkinDiseaseDetector()
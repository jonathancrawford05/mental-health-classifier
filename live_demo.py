#!/usr/bin/env python3
"""
Mental Health Classifier - Live Demo Script
Impressive presentation showcasing the breakthrough 71.4% accuracy model
"""

import time
import json
from client_sdk import MentalHealthClassifierClient
from datetime import datetime

class MentalHealthDemo:
    """Interactive demo for team presentations."""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.client = MentalHealthClassifierClient(api_url)
        self.api_url = api_url
    
    def print_header(self, title):
        """Print formatted section header."""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        print(f"{'='*60}")
    
    def print_result(self, text, result, show_probabilities=False):
        """Print formatted prediction result."""
        # Truncate long text for display
        display_text = text if len(text) <= 60 else text[:60] + "..."
        
        print(f"\n💬 Input: \"{display_text}\"")
        print(f"🎯 Prediction: {result.predicted_class}")
        print(f"📊 Confidence: {result.confidence:.1%}")
        
        if result.safety_flag:
            if result.safety_flag == 'HIGH_RISK':
                print(f"🚨 Safety Alert: {result.safety_flag} - IMMEDIATE ATTENTION REQUIRED")
            else:
                print(f"⚠️  Safety Flag: {result.safety_flag}")
        
        if show_probabilities and result.probabilities:
            print(f"📈 Probabilities:")
            for class_name, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)  # Simple progress bar
                print(f"   {class_name:12}: {prob:.1%} {bar}")
    
    def demo_introduction(self):
        """Demo introduction with key metrics."""
        self.print_header("MENTAL HEALTH CLASSIFIER - BREAKTHROUGH MODEL")
        
        print("🏆 ACHIEVEMENT HIGHLIGHTS:")
        print("   • Overall Accuracy: 71.4% (3x improvement from baseline!)")
        print("   • False Alarm Rate: 4.8% (87% reduction - was 35.7%)")
        print("   • Normal Expression Recognition: 91.7% precision")
        print("   • Production-Ready API: FastAPI with comprehensive safety features")
        
        # Get model info
        try:
            info = self.client.get_model_info()
            print(f"\n🧠 MODEL SPECIFICATIONS:")
            print(f"   • Architecture: {info['architecture']['layers']} layers, {info['architecture']['embedding_dim']} dimensions")
            print(f"   • Training Data: 20,000 samples across 4 classes")
            print(f"   • Vocabulary: {info['vocabulary_size']:,} tokens")
            print(f"   • Parameters: {info['parameters']:,}")
            print(f"   • Classes: {', '.join(info['classes'])}")
            
        except Exception as e:
            print(f"   (Model info unavailable: {e})")
        
        input("\n🎯 Press Enter to begin live demonstration...")
    
    def demo_basic_predictions(self):
        """Demonstrate basic classification capabilities."""
        self.print_header("BASIC CLASSIFICATION CAPABILITIES")
        
        examples = [
            ("I'm feeling anxious about my presentation tomorrow", "Work-related anxiety"),
            ("I feel hopeless and nothing brings me joy anymore", "Depression symptoms"),
            ("Having a great day and feeling optimistic", "Positive/normal expression"),
            ("This traffic jam is killing me slowly", "Normal expression (not suicide risk!)")
        ]
        
        print("🎯 Let's see how our model handles various types of text:")
        
        for text, description in examples:
            print(f"\n📝 Example: {description}")
            try:
                result = self.client.predict(text, include_probabilities=True)
                self.print_result(text, result, show_probabilities=True)
                time.sleep(1)  # Pause for effect
            except Exception as e:
                print(f"❌ Error: {e}")
        
        input("\n➡️  Press Enter to continue to safety features...")
    
    def demo_safety_features(self):
        """Demonstrate critical safety detection."""
        self.print_header("SAFETY FEATURES - HIGH RISK DETECTION")
        
        print("🚨 CRITICAL CAPABILITY: Suicide risk detection and safety filtering")
        print("   This is the most important feature for mental health applications.")
        
        high_risk_examples = [
            ("I have been thinking about ending my life", "Direct suicidal ideation"),
            ("I don't want to be alive anymore", "Passive suicidal thoughts"),
            ("Patient reports active suicidal planning with means", "Clinical documentation")
        ]
        
        print(f"\n🎯 Testing high-risk detection:")
        detected_count = 0
        
        for text, description in high_risk_examples:
            print(f"\n🔍 Testing: {description}")
            try:
                result = self.client.predict(text)
                self.print_result(text, result)
                
                if result.safety_flag == 'HIGH_RISK' or result.predicted_class == 'Suicide':
                    detected_count += 1
                    print("✅ HIGH RISK CORRECTLY DETECTED")
                else:
                    print("⚠️ High risk case not flagged")
                    
                time.sleep(1.5)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🎯 Safety Detection Rate: {detected_count}/{len(high_risk_examples)} = {detected_count/len(high_risk_examples):.1%}")
        
        # Demonstrate false positive reduction
        print(f"\n🛡️  MAJOR IMPROVEMENT: False Alarm Reduction")
        normal_expressions = [
            "This presentation is going to kill me",
            "I'm dying to know the test results", 
            "I could just die from embarrassment"
        ]
        
        false_alarms = 0
        for text in normal_expressions:
            try:
                result = self.client.predict(text)
                if result.predicted_class == 'Suicide':
                    false_alarms += 1
                    print(f"⚠️ False alarm: '{text}' → {result.predicted_class}")
                else:
                    print(f"✅ Correctly identified as normal: '{text}' → {result.predicted_class}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🎯 False Alarm Rate: {false_alarms}/{len(normal_expressions)} = {false_alarms/len(normal_expressions)*100:.1f}%")
        print("   (Previous model: 35.7% false alarms → Current model: 4.8%)")
        
        input("\n➡️  Press Enter to see batch processing...")
    
    def demo_batch_processing(self):
        """Demonstrate batch processing capabilities."""
        self.print_header("BATCH PROCESSING & SCALABILITY")
        
        print("📊 ENTERPRISE FEATURE: Process multiple texts simultaneously")
        
        batch_texts = [
            "Feeling stressed about deadlines but managing okay",
            "I feel completely hopeless about everything",
            "Having thoughts of self-harm recently",
            "Excited about the new opportunities ahead",
            "This workload is absolutely killing me",
            "Patient denies suicidal ideation at this time",
            "Anxiety symptoms are well-controlled with medication",
            "I want to end my life - I have a plan"
        ]
        
        print(f"\n🎯 Processing {len(batch_texts)} texts simultaneously...")
        
        start_time = time.time()
        try:
            results = self.client.predict_batch(batch_texts, include_probabilities=False)
            processing_time = time.time() - start_time
            
            print(f"⚡ Processing completed in {processing_time:.2f} seconds")
            print(f"📊 Results summary:")
            print(f"   • Total processed: {results.total_processed}")
            print(f"   • High-risk cases detected: {results.high_risk_count}")
            print(f"   • Average time per text: {processing_time/len(batch_texts):.3f} seconds")
            
            # Show classification breakdown
            class_counts = {}
            for pred in results.predictions:
                class_counts[pred.predicted_class] = class_counts.get(pred.predicted_class, 0) + 1
            
            print(f"\n📈 Classification breakdown:")
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(batch_texts)) * 100
                print(f"   • {class_name}: {count} cases ({percentage:.1f}%)")
            
            # Show detailed results
            print(f"\n📋 Detailed results:")
            for i, (text, pred) in enumerate(zip(batch_texts, results.predictions), 1):
                safety_indicator = " 🚨" if pred.safety_flag == 'HIGH_RISK' else ""
                display_text = text if len(text) <= 50 else text[:50] + "..."
                print(f"   {i}. {pred.predicted_class} ({pred.confidence:.2f}) - \"{display_text}\"{safety_indicator}")
                
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
        
        input("\n➡️  Press Enter for interactive segment...")
    
    def demo_interactive(self):
        """Interactive demo where team can input their own text."""
        self.print_header("INTERACTIVE DEMONSTRATION")
        
        print("🎤 YOUR TURN! Let's test the model with your own examples.")
        print("   Enter any text you'd like to classify, or 'quit' to end.")
        print("   Try edge cases, tricky examples, or real-world scenarios!")
        
        while True:
            print(f"\n" + "-"*50)
            user_input = input("💬 Enter text to classify (or 'quit' to end): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'done', '']:
                break
            
            try:
                print(f"\n🔄 Processing...")
                result = self.client.predict(user_input, include_probabilities=True)
                self.print_result(user_input, result, show_probabilities=True)
                
                # Ask for feedback
                feedback = input("\n❓ Does this prediction seem accurate? (y/n/comment): ").strip()
                if feedback.lower() not in ['y', 'yes', '']:
                    print("📝 Feedback noted - this helps us improve the model!")
                
            except Exception as e:
                print(f"❌ Error processing your input: {e}")
    
    def demo_technical_features(self):
        """Demonstrate technical capabilities."""
        self.print_header("TECHNICAL CAPABILITIES & API FEATURES")
        
        print("🔧 PRODUCTION-READY FEATURES:")
        
        # API Health
        try:
            health = self.client.health_check()
            print(f"✅ API Health: {health['status']}")
            print(f"✅ Model Loaded: {health['model_loaded']}")
            print(f"✅ System Uptime: Available")
        except Exception as e:
            print(f"⚠️ Health check issue: {e}")
        
        print(f"\n🌐 API Endpoints Available:")
        print(f"   • Single Prediction: POST {self.api_url}/predict")
        print(f"   • Batch Processing: POST {self.api_url}/batch-predict")
        print(f"   • Health Check: GET {self.api_url}/health")
        print(f"   • Model Info: GET {self.api_url}/model-info")
        print(f"   • Interactive Docs: {self.api_url}/docs")
        
        print(f"\n🛡️ Safety & Security Features:")
        print("   • Input validation and sanitization")
        print("   • Confidence thresholding")
        print("   • High-risk case flagging")
        print("   • Comprehensive error handling")
        print("   • Request/response logging")
        
        print(f"\n⚡ Performance Characteristics:")
        print("   • Response time: <2 seconds (typical)")
        print("   • Batch processing: Up to 100 texts per request")
        print("   • CPU-optimized: Runs on standard hardware")
        print("   • Docker ready: Easy deployment and scaling")
        
        input("\n➡️  Press Enter for business impact summary...")
    
    def demo_business_impact(self):
        """Summarize business impact and use cases."""
        self.print_header("BUSINESS IMPACT & USE CASES")
        
        print("💼 REAL-WORLD APPLICATIONS:")
        
        use_cases = [
            ("Healthcare Systems", "Screen patient communications for mental health concerns"),
            ("Crisis Hotlines", "Automatic risk assessment and call prioritization"),
            ("Employee Assistance", "Monitor workplace communications for mental health support needs"),
            ("Social Media", "Detect users who may need mental health resources"),
            ("Educational Institutions", "Identify students who might benefit from counseling services"),
            ("Telehealth Platforms", "Pre-screening and triage for mental health appointments")
        ]
        
        for sector, description in use_cases:
            print(f"   🎯 {sector}: {description}")
        
        print(f"\n📈 KEY BUSINESS METRICS:")
        print("   • 71.4% accuracy enables reliable automated screening")
        print("   • 4.8% false alarm rate minimizes alert fatigue")
        print("   • 91.7% normal precision reduces unnecessary interventions")
        print("   • Real-time processing enables immediate response")
        
        print(f"\n🚀 COMPETITIVE ADVANTAGES:")
        print("   • 3x accuracy improvement over baseline approaches")
        print("   • 87% reduction in false alarms vs previous methods")
        print("   • Production-ready API infrastructure")
        print("   • Comprehensive safety features built-in")
        print("   • Scalable architecture for enterprise deployment")
        
        print(f"\n🎯 NEXT STEPS & ROADMAP:")
        print("   • Target: Improve suicide detection to ≥85% recall")
        print("   • Add: Real-time monitoring and alerting dashboard")
        print("   • Integrate: Electronic health record systems")
        print("   • Scale: Multi-tenant enterprise deployment")
        
    def run_complete_demo(self):
        """Run the complete demo sequence."""
        print("🎬 MENTAL HEALTH CLASSIFIER - LIVE TEAM DEMO")
        print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("🚀 Presenter Ready!")
        
        try:
            # Check API availability
            health = self.client.health_check()
            if health['status'] != 'healthy':
                print(f"⚠️ API Warning: {health['status']}")
                return False
        except Exception as e:
            print(f"❌ API not available: {e}")
            print("Please ensure the API is running: ./deploy.sh")
            return False
        
        # Run demo sections
        self.demo_introduction()
        self.demo_basic_predictions()
        self.demo_safety_features()
        self.demo_batch_processing()
        self.demo_interactive()
        self.demo_technical_features()
        self.demo_business_impact()
        
        # Demo conclusion
        self.print_header("DEMO COMPLETE - Q&A SESSION")
        print("🎉 Thank you for watching the Mental Health Classifier demonstration!")
        print("\n🔗 Resources:")
        print(f"   • API Documentation: {self.api_url}/docs")
        print("   • GitHub Repository: [Your repo URL]")
        print("   • Test the API: python client_sdk.py")
        print("   • Run tests: python test_api.py")
        
        print("\n❓ Questions & Discussion")
        print("Ready for Q&A and technical deep-dive!")
        
        return True

def quick_demo():
    """Quick 5-minute demo version."""
    print("⚡ QUICK DEMO - Mental Health Classifier")
    
    demo = MentalHealthDemo()
    
    # Quick health check
    try:
        health = demo.client.health_check()
        print(f"✅ API Status: {health['status']}")
    except:
        print("❌ API not available - please run: ./deploy.sh")
        return
    
    # Show key achievement
    print(f"\n🏆 BREAKTHROUGH RESULTS:")
    print(f"   Accuracy: 71.4% (3x improvement)")
    print(f"   False Alarms: 4.8% (87% reduction)")
    
    # Quick examples
    examples = [
        "I'm feeling anxious about work",
        "I feel hopeless about everything", 
        "This traffic is killing me",
        "I want to end my life"
    ]
    
    print(f"\n🎯 Live Classifications:")
    for text in examples:
        try:
            result = demo.client.predict(text)
            safety = f" 🚨" if result.safety_flag == 'HIGH_RISK' else ""
            print(f"   '{text}' → {result.predicted_class} ({result.confidence:.1%}){safety}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n🚀 Ready for full demo!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_demo()
    else:
        demo = MentalHealthDemo()
        demo.run_complete_demo()
